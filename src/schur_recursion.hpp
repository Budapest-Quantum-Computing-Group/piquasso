/*
 * Copyright 2021-2026 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PIQUASSO_SCHUR_RECURSION_HPP
#define PIQUASSO_SCHUR_RECURSION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "matching.hpp"
#include "matrix.hpp"
#include "neumaier.hpp"
#include "polynomial.hpp"
#include "utils.hpp"

// Finite-difference Schur recursion for hafnians and loop hafnians with repeated
// rows. The idea behind this algorithm will be described in a forthcoming
// arXiv preprint.
//
// Every level eliminates one edge of the matching of the photons by a 2x2 Schur
// complement over polynomials in t modulo t^(n+1), for every Glynn term of the
// edge, and accumulates the logarithms of the pivot factors in an exponent. A
// leaf adds [t^n] exp(exponent).
//
// Notation at a level, in blocks of the pair to eliminate (1) and the rest (2):
//   A11     [[a, b], [b, c]], with A_p0 and A_p1 the entries of row p of A21,
//   mu, m   the loop weights and, in a batch, the marked couplings,
//   z       the signed count of a Glynn term,
//   E       the pivot z X_2 (I - t z A11 X_2)^-1, with X_2 = [[0, 1], [1, 0]],
//   det     det(I - t z A11 X_2) = 1 - 2 z t b + z^2 t^2 S, with S = b^2 - ac.
// The next level holds A22 + t A21 E A12, and similarly mu and m.
//
// Only [t^n] of the exponent is needed, so a quantity that enters it multiplied
// by t^k needs only its first n + 1 - k coefficients: A, mu, m and beta enter
// with t and need n, S, Y and U with t^2 and need n - 1, and X with t^3 and
// needs n - 2.

namespace piquasso::hafnian {

using namespace piquasso::polynomial;

using Complex = std::complex<double>;

struct GlynnTerm {
    double signed_count = 0;
    long double weight = 0;
};

using GlynnTerms = std::vector<std::vector<GlynnTerm>>;

// The Glynn terms of an edge of multiplicity eta: signed counts z = eta - 2k with
// weights (-1)^k binom(eta, k) / 2^eta. With `root_symmetry`, only z >= 0 is kept
// and z > 0 is doubled.
inline std::vector<GlynnTerm> make_glynn_terms(
    std::size_t eta,
    bool root_symmetry
) {
    if (eta == 0)
        throw std::logic_error("edge of zero multiplicity");

    const std::vector<long double> probabilities =
        binomialProbabilities(static_cast<std::int64_t>(eta));

    std::vector<GlynnTerm> terms;
    for (std::size_t k = 0; k <= eta; ++k) {
        const double z = static_cast<double>(eta) - 2.0 * static_cast<double>(k);
        if (root_symmetry && z < 0.0)
            continue;

        long double weight = (k % 2 ? -1.0L : 1.0L) * probabilities[k];
        if (root_symmetry && z > 0.0)
            weight *= 2.0L;

        terms.push_back({z, weight});
    }

    return terms;
}

// The exponent s for which pair weights divided by 4^s and loop weights divided
// by 2^s are of order one. This divides the loop hafnian of N vertices by
// 2^(sN) exactly.
inline int scale_exponent(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal
) {
    double largest = 0.0;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        largest = std::max(largest, std::abs(diagonal[row]));

        for (std::size_t col = 0; col < matrix.cols; ++col)
            largest = std::max(largest, std::sqrt(std::abs(matrix(row, col))));
    }
    if (largest > 0.0 && std::isfinite(largest))
        return static_cast<int>(std::lround(std::log2(largest)));

    return 0;
}

inline Complex rescale(const Complex &value, int exponent) {
    return {std::ldexp(value.real(), exponent), std::ldexp(value.imag(), exponent)};
}

struct SchurProblem {
    std::size_t n = 0; // Truncation order, half the number of vertices.
    std::size_t photons = 0; // Of the input, i.e., without the dummy mode.
    Matrix<Complex> matrix;
    Vector<Complex> mu;
    GlynnTerms glynn_terms; // For every edge of the matching, i.e., level.
    bool track_source_terms = true; // The loop weights mu, zero for the hafnian.
    int scale_exponent = 0;

    // Batch over 0, ..., outputs-1 copies of a marked vertex.
    std::size_t outputs = 1;
    Vector<Complex> marked_couplings;
    Complex marked_pair = 0.0;
    Complex marked_loop = 0.0;
};

inline bool is_batch(const SchurProblem &problem) { return problem.outputs > 1; }

// The exponent has components 1, x and x^2 in the weight x of the marked vertex.
inline std::size_t components(const SchurProblem &problem) {
    return is_batch(problem) ? 3 : 1;
}

// The quadratic forms of the exponent, see `quadratic_form`.
inline std::size_t forms(const SchurProblem &problem) {
    return problem.track_source_terms ? components(problem) : 0;
}

// The input scaled by `scale_exponent`, with rows and columns in the order of
// the edges of the matching. The marked vertex of a batch is the last mode.
//
// A dummy mode with one photon is prepended for an odd number of photons. Its
// loop weight is one after scaling, not before: as 2^-s, it would make the
// result a cancellation of terms 2^|s| times larger.
inline SchurProblem make_problem(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal,
    const std::vector<std::int64_t> &occupations,
    bool track_source_terms,
    std::size_t outputs = 1
) {
    const std::int64_t photons =
        std::accumulate(occupations.begin(), occupations.end(), std::int64_t{0});
    const std::size_t pad = photons % 2 != 0 ? 1 : 0;
    std::vector<std::int64_t> padded(pad, 1);
    padded.insert(padded.end(), occupations.begin(), occupations.end());

    SchurProblem problem;
    problem.photons = static_cast<std::size_t>(photons);
    problem.n = (problem.photons + pad) / 2;
    problem.track_source_terms = track_source_terms;
    problem.outputs = outputs;
    problem.scale_exponent = scale_exponent(matrix, diagonal);

    const auto pair_weight = [&](std::size_t i, std::size_t j) {
        return i < pad || j < pad
            ? Complex{}
            : rescale(matrix(i - pad, j - pad), -2 * problem.scale_exponent);
    };
    const auto loop_weight = [&](std::size_t i) {
        return i < pad
            ? Complex(1.0)
            : rescale(diagonal[i - pad], -problem.scale_exponent);
    };

    // Increasing multiplicity is the cheapest order of elimination.
    Matching matching = match_occupation_numbers(padded);
    std::stable_sort(
        matching.begin(), matching.end(),
        [](const MatchingEdge &left, const MatchingEdge &right) {
            return left.multiplicity < right.multiplicity;
        }
    );

    std::vector<std::size_t> modes;
    for (const MatchingEdge &edge : matching) {
        modes.push_back(edge.first);
        modes.push_back(edge.second);
        problem.glynn_terms.push_back(
            make_glynn_terms(edge.multiplicity, problem.glynn_terms.empty())
        );
    }

    const std::size_t dim = modes.size();
    problem.matrix = Matrix<Complex>(dim, dim);
    problem.mu = Vector<Complex>(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        problem.mu[i] = loop_weight(modes[i]);
        for (std::size_t j = 0; j < dim; ++j)
            problem.matrix(i, j) = pair_weight(modes[i], modes[j]);
    }

    if (is_batch(problem)) {
        const std::size_t last = matrix.rows - 1 + pad;
        problem.marked_couplings = Vector<Complex>(dim);
        for (std::size_t i = 0; i < dim; ++i)
            problem.marked_couplings[i] = pair_weight(modes[i], last);

        problem.marked_pair = pair_weight(last, last);
        problem.marked_loop = loop_weight(last);
    }

    return problem;
}

// The state of a level, followed by the parts of the pivot shared by all Glynn
// terms, with the number of coefficients used.
struct SchurLevel {
    std::size_t dim = 0;              // The number of remaining vertices.
    PolynomialArray A;                // Packed upper triangle: n.
    PolynomialArray mu;               // n.
    PolynomialArray marked_couplings; // m: n.
    PolynomialArray exponent;         // Parts of order 1, x and x^2: n + 1.
    PolynomialArray S;                // b^2 - ac: n - 1.
    PolynomialArray beta;             // x0 y1 + x1 y0 for every form: n.
    PolynomialArray Y;                // c x0 y0 + a x1 y1 - b beta: n - 1.
    PolynomialArray X0;               // A_p0 c - A_p1 b for every row p: n - 2.
    PolynomialArray X1;               // A_p1 a - A_p0 b: n - 2.
};

// The vectors and the factor of a quadratic form f x^T E y, which enters the
// exponent as t f x^T E y.
struct QuadraticForm {
    const PolynomialArray &x;
    const PolynomialArray &y;
    double factor;
};

// The parts of the exponent of order 1, x and x^2 in the weight x of the marked
// vertex: (1/2) mu^T E mu, mu^T E m and (1/2) m^T E m.
inline QuadraticForm quadratic_form(const SchurLevel &state, std::size_t form) {
    switch (form) {
    case 0:
        return {state.mu, state.mu, 0.5};
    case 1:
        return {state.mu, state.marked_couplings, 1.0};
    default:
        return {state.marked_couplings, state.marked_couplings, 0.5};
    }
}

inline std::size_t packed_index(std::size_t dim, std::size_t i, std::size_t j) {
    return i * (2 * dim - i + 1) / 2 + (j - i);
}

inline Polynomial entry(SchurLevel &state, std::size_t i, std::size_t j) {
    return state.A[packed_index(state.dim, i, j)];
}

inline ConstPolynomial entry(
    const SchurLevel &state,
    std::size_t i,
    std::size_t j
) {
    return state.A[packed_index(state.dim, i, j)];
}

// The buffers of the depth-first recursion of one thread, with the number of
// coefficients used.
struct Workspace {
    std::vector<SchurLevel> levels;
    PolynomialArray det;           // n + 1.
    PolynomialArray inverse;       // 1 / det: n.
    PolynomialArray log_factor;    // The log of the pivot factors: n + 1.
    PolynomialArray scratch;       // n.
    PolynomialArray U0;            // U = A21 E, for every row p: n - 1.
    PolynomialArray U1;            // n - 1.
    PolynomialArray leaf_exponent; // n + 1.
    PolynomialArray R;             // The batch recurrence: n + 1.
};

// Allocates the levels from `first_level` on. The root level, if allocated,
// holds the input.
inline Workspace make_workspace(
    const SchurProblem &problem,
    std::size_t first_level
) {
    const std::size_t n = problem.n;
    const std::size_t pairs = problem.glynn_terms.size();
    Workspace workspace;
    workspace.levels.resize(pairs);
    for (std::size_t level = first_level; level < pairs; ++level) {
        const std::size_t dim = 2 * (pairs - level);
        SchurLevel &state = workspace.levels[level];
        state.dim = dim;
        state.A = PolynomialArray(dim * (dim + 1) / 2, n);
        state.mu = PolynomialArray(problem.track_source_terms ? dim : 0, n);
        state.marked_couplings = PolynomialArray(is_batch(problem) ? dim : 0, n);
        state.exponent = PolynomialArray(components(problem), n + 1);
        state.S = PolynomialArray(1, n);
        state.beta = PolynomialArray(forms(problem), n);
        state.Y = PolynomialArray(forms(problem), n);
        state.X0 = PolynomialArray(dim - 2, n);
        state.X1 = PolynomialArray(dim - 2, n);
    }
    const std::size_t rows = 2 * (pairs - first_level) - 2;
    workspace.det = PolynomialArray(1, n + 1);
    workspace.inverse = PolynomialArray(1, n);
    workspace.log_factor = PolynomialArray(components(problem), n + 1);
    workspace.scratch = PolynomialArray(2, n);
    workspace.U0 = PolynomialArray(rows, n);
    workspace.U1 = PolynomialArray(rows, n);
    workspace.leaf_exponent = PolynomialArray(components(problem), n + 1);
    workspace.R = PolynomialArray(3, n + 1);

    if (first_level == 0) {
        SchurLevel &root = workspace.levels[0];
        for (std::size_t i = 0; i < root.dim; ++i) {
            for (std::size_t j = i; j < root.dim; ++j)
                set_coefficient(entry(root, i, j), 0, problem.matrix(i, j));

            if (problem.track_source_terms)
                set_coefficient(root.mu[i], 0, problem.mu[i]);

            if (is_batch(problem)) {
                set_coefficient(root.marked_couplings[i], 0,
                                problem.marked_couplings[i]);
            }
        }
    }

    return workspace;
}

// The parts of the pivot shared by all Glynn terms of `level`: S, beta and Y,
// for which x^T E y = (z beta + z^2 t Y) / det, and X0 and X1, for which
// A21 E = (z (A_p1, A_p0) + z^2 t (X0, X1)) / det.
inline void prepare_pivot(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level
) {
    SchurLevel &state = workspace.levels[level];
    const std::size_t n = problem.n;
    const ConstPolynomial a = entry(state, 0, 0);
    const ConstPolynomial b = entry(state, 0, 1);
    const ConstPolynomial c = entry(state, 1, 1);

    product_difference(state.S[0], b, b, a, c, n - 1);

    const Polynomial xy00 = workspace.scratch[0];
    const Polynomial xy11 = workspace.scratch[1];
    for (std::size_t form = 0; form < forms(problem); ++form) {
        const QuadraticForm quadratic = quadratic_form(state, form);
        const PolynomialArray &x = quadratic.x;
        const PolynomialArray &y = quadratic.y;
        const Polynomial beta = state.beta[form];
        const Polynomial Y = state.Y[form];
        product_sum(beta, x[0], y[1], x[1], y[0], n);
        product(xy00, x[0], y[0], n - 1);
        product(xy11, x[1], y[1], n - 1);
        product_sum(Y, c, xy00, a, xy11, n - 1);
        mul_add(Y, b, beta, n - 1, 0, -1.0);
    }

    for (std::size_t p = 0; p + 2 < state.dim; ++p) {
        const ConstPolynomial Ap0 = entry(state, 0, p + 2);
        const ConstPolynomial Ap1 = entry(state, 1, p + 2);
        product_difference(state.X0[p], Ap0, c, Ap1, b, n - 2);
        product_difference(state.X1[p], Ap1, a, Ap0, b, n - 2);
    }
}

// destination = exponent - (1/2) log det + t sum f x^T E y over the forms, for
// the signed count z. Leaves 1 / det in workspace.inverse.
inline void pivot(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level,
    double z,
    PolynomialArray &destination
) {
    const SchurLevel &state = workspace.levels[level];
    const std::size_t n = problem.n;
    // E = 0 and det = 1.
    if (z == 0.0) {
        for (std::size_t component = 0; component < components(problem);
             ++component) {
            copy(destination[component], state.exponent[component], n + 1);
        }
        return;
    }

    // det = 1 + t (-2 z b + z^2 t S)
    const Polynomial det = workspace.det[0];
    const Polynomial inverse = workspace.inverse[0];
    set_coefficient(det, 0, 1.0);
    combine(from_degree(det, 1), -2.0 * z, entry(state, 0, 1), z * z,
            state.S[0], n);
    inverse_unit(inverse, det, n);

    PolynomialArray &log_factor = workspace.log_factor;
    log_unit(log_factor[0], det, inverse, n, -0.5, workspace.scratch[0]);
    for (std::size_t component = 1; component < components(problem);
         ++component) {
        set_zero(log_factor[component], n + 1);
    }
    const Polynomial numerator = workspace.scratch[0];
    for (std::size_t form = 0; form < forms(problem); ++form) {
        combine(numerator, z, state.beta[form], z * z, state.Y[form], n);
        mul_add(log_factor[form], numerator, inverse, n + 1, 1,
                quadratic_form(state, form).factor);
    }

    for (std::size_t component = 0; component < components(problem); ++component) {
        add(destination[component], state.exponent[component],
            log_factor[component], n + 1);
    }
}

// The next level for z = 0, where E = 0: the rest of the parent, unchanged.
inline void copy_rest(
    const SchurProblem &problem,
    const SchurLevel &parent,
    SchurLevel &child
) {
    const std::size_t n = problem.n;
    for (std::size_t i = 0; i + 2 < parent.dim; ++i) {
        for (std::size_t j = i; j + 2 < parent.dim; ++j)
            copy(entry(child, i, j), entry(parent, i + 2, j + 2), n);

        if (problem.track_source_terms)
            copy(child.mu[i], parent.mu[i + 2], n);

        if (is_batch(problem))
            copy(child.marked_couplings[i], parent.marked_couplings[i + 2], n);
    }
}

// Eliminates the pair of `level` with the signed count z: the next level holds
// A22 + t A21 E A12, and similarly mu and m.
inline void eliminate_pair(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level,
    double z
) {
    const SchurLevel &parent = workspace.levels[level];
    SchurLevel &child = workspace.levels[level + 1];
    const std::size_t n = problem.n;
    const std::size_t rows = parent.dim - 2;

    pivot(problem, workspace, level, z, child.exponent);
    if (z == 0.0) {
        copy_rest(problem, parent, child);
        return;
    }

    // U = A21 E. X_2 in E swaps the columns, so U0 = (z A_p1 + z^2 t X0) / det
    // and U1 = (z A_p0 + z^2 t X1) / det.
    PolynomialArray &U0 = workspace.U0;
    PolynomialArray &U1 = workspace.U1;
    const Polynomial numerator = workspace.scratch[0];
    const auto divide = [&](Polynomial out, ConstPolynomial x,
                            ConstPolynomial y) {
        combine(numerator, z, x, z * z, y, n - 1);
        product(out, numerator, workspace.inverse[0], n - 1);
    };
    for (std::size_t p = 0; p < rows; ++p) {
        divide(U0[p], entry(parent, 1, p + 2), parent.X0[p]);
        divide(U1[p], entry(parent, 0, p + 2), parent.X1[p]);
    }

    // target = value + t (U0_i x0 + U1_i x1)
    const auto update = [&](Polynomial target, ConstPolynomial value,
                            std::size_t i, ConstPolynomial x0,
                            ConstPolynomial x1) {
        copy(target, value, n);
        mul_add(target, U0[i], x0, n, 1);
        mul_add(target, U1[i], x1, n, 1);
    };
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = i; j < rows; ++j) {
            update(entry(child, i, j), entry(parent, i + 2, j + 2), i,
                   entry(parent, 0, j + 2), entry(parent, 1, j + 2));
        }
        if (problem.track_source_terms)
            update(child.mu[i], parent.mu[i + 2], i, parent.mu[0], parent.mu[1]);

        if (is_batch(problem)) {
            update(child.marked_couplings[i], parent.marked_couplings[i + 2], i,
                   parent.marked_couplings[0], parent.marked_couplings[1]);
        }
    }
}

// Adds weight [t^n] R_k for every output k, where R_0 = exp(exponent) and
// k R_k = L R_(k-1) + 2 Q R_(k-2) for the copies of the marked vertex. L and Q
// are the parts of the exponent linear and quadratic in its weight x, plus its
// own loop and pair weights.
inline void leaf(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level,
    double z,
    double weight,
    NeumaierSum<Complex> *sums
) {
    const std::size_t n = problem.n;
    PolynomialArray &exponent = workspace.leaf_exponent;
    pivot(problem, workspace, level, z, exponent);

    Polynomial previous2 = workspace.R[0];
    Polynomial previous = workspace.R[1];
    Polynomial current = workspace.R[2];
    exponential(previous, exponent[0], n, workspace.scratch[0]);
    compensated_add(sums[0], weight * coefficient(previous, n));
    if (!is_batch(problem))
        return;

    const Polynomial linear = exponent[1];
    const Polynomial quadratic = exponent[2];
    set_coefficient(linear, 0, coefficient(linear, 0) + problem.marked_loop);
    set_coefficient(quadratic, 0,
                    coefficient(quadratic, 0) + 0.5 * problem.marked_pair);
    for (std::size_t k = 1; k < problem.outputs; ++k) {
        const double inverse_k = 1.0 / static_cast<double>(k);
        set_zero(current, n + 1);
        mul_add(current, linear, previous, n + 1, 0, inverse_k);
        if (k >= 2)
            mul_add(current, quadratic, previous2, n + 1, 0, 2.0 * inverse_k);

        compensated_add(sums[k], weight * coefficient(current, n));
        std::swap(previous2, previous);
        std::swap(previous, current);
    }
}

// Adds the leaves below the Glynn terms [first, past) of `level`.
inline void evaluate_subtree(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level,
    std::size_t first,
    std::size_t past,
    long double weight,
    NeumaierSum<Complex> *sums
) {
    const std::vector<GlynnTerm> &terms = problem.glynn_terms[level];
    prepare_pivot(problem, workspace, level);
    for (std::size_t k = first; k < past; ++k) {
        const long double child_weight = weight * terms[k].weight;
        if (level + 1 == problem.glynn_terms.size()) {
            leaf(problem, workspace, level, terms[k].signed_count,
                 static_cast<double>(child_weight), sums);
        } else {
            eliminate_pair(problem, workspace, level, terms[k].signed_count);
            evaluate_subtree(problem, workspace, level + 1, 0,
                             problem.glynn_terms[level + 1].size(),
                             child_weight, sums);
        }
    }
}

// The Glynn terms [first_term, past_term) of a level, below a fixed path.
struct Task {
    std::size_t first_term = 0;
    std::size_t past_term = 0;
    long double weight = 1.0L;
    SchurLevel state;
};

// Expands the tree down to `cut`, splitting the Glynn terms there into `chunks`.
inline void collect(
    const SchurProblem &problem,
    Workspace &workspace,
    std::size_t level,
    std::size_t cut,
    std::size_t chunks,
    long double weight,
    std::vector<Task> &tasks
) {
    const std::vector<GlynnTerm> &terms = problem.glynn_terms[level];
    if (level == cut) {
        const std::size_t chunk_size = (terms.size() + chunks - 1) / chunks;
        for (std::size_t k = 0; k < terms.size(); k += chunk_size) {
            tasks.push_back({k, std::min(k + chunk_size, terms.size()), weight,
                             workspace.levels[level]});
        }
        return;
    }
    prepare_pivot(problem, workspace, level);
    for (const GlynnTerm &term : terms) {
        eliminate_pair(problem, workspace, level, term.signed_count);
        collect(problem, workspace, level + 1, cut, chunks,
                weight * term.weight, tasks);
    }
}

// Independent of the number of threads, so that the result is too.
constexpr std::size_t kTaskCount = 64;

constexpr long double kParallelWorkThreshold = 1.0e4L;

// Evaluates about kTaskCount subtrees in parallel, and adds them in a fixed
// order.
inline void evaluate_in_parallel(
    const SchurProblem &problem,
    Workspace &root,
    std::vector<NeumaierSum<Complex>> &sums
) {
    const std::size_t outputs = problem.outputs;

    std::size_t cut = 0;
    std::size_t subtrees = 1;
    while (cut + 1 < problem.glynn_terms.size()
           && subtrees * problem.glynn_terms[cut].size() < kTaskCount) {
        subtrees *= problem.glynn_terms[cut].size();
        ++cut;
    }
    const std::size_t chunks = std::min(
        problem.glynn_terms[cut].size(), (kTaskCount + subtrees - 1) / subtrees
    );
    std::vector<Task> tasks;
    collect(problem, root, 0, cut, chunks, 1.0L, tasks);

#if defined(_OPENMP)
    const std::size_t threads =
        std::min(static_cast<std::size_t>(omp_get_max_threads()), tasks.size());
#else
    const std::size_t threads = 1;
#endif
    std::vector<Workspace> workspaces(threads, make_workspace(problem, cut));
    std::vector<NeumaierSum<Complex>> partial(tasks.size() * outputs);
    const auto task_count = static_cast<std::int64_t>(tasks.size());

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) num_threads(static_cast<int>(threads)) if(threads > 1)
#endif
    for (std::int64_t index = 0; index < task_count; ++index) {
#if defined(_OPENMP)
        Workspace &workspace =
            workspaces[static_cast<std::size_t>(omp_get_thread_num())];
#else
        Workspace &workspace = workspaces[0];
#endif
        const auto t = static_cast<std::size_t>(index);
        workspace.levels[cut] = tasks[t].state;
        evaluate_subtree(problem, workspace, cut, tasks[t].first_term,
                         tasks[t].past_term, tasks[t].weight,
                         &partial[t * outputs]);
    }

    for (std::size_t t = 0; t < tasks.size(); ++t) {
        for (std::size_t k = 0; k < outputs; ++k)
            compensated_add(sums[k], compensated_value(partial[t * outputs + k]));
    }
}

// The loop hafnians with 0, ..., outputs-1 copies of the marked vertex, each
// divided by the factorial of the number of copies.
inline std::vector<Complex> evaluate_problem(const SchurProblem &problem) {
    const std::size_t outputs = problem.outputs;

    long double work = static_cast<long double>(problem.n)
        * static_cast<long double>(problem.n) * static_cast<long double>(outputs);
    for (const std::vector<GlynnTerm> &terms : problem.glynn_terms)
        work *= static_cast<long double>(terms.size());

    Workspace root = make_workspace(problem, 0);
    std::vector<NeumaierSum<Complex>> sums(outputs);
    if (work < kParallelWorkThreshold) {
        evaluate_subtree(problem, root, 0, 0, problem.glynn_terms[0].size(),
                         1.0L, sums.data());
    } else {
        evaluate_in_parallel(problem, root, sums);
    }

    std::vector<Complex> result(outputs);
    for (std::size_t k = 0; k < outputs; ++k) {
        result[k] = rescale(
            compensated_value(sums[k]),
            problem.scale_exponent * static_cast<int>(problem.photons + k)
        );
    }

    return result;
}

} // namespace piquasso::hafnian

#endif
