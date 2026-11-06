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
#include "polynomial.hpp"
#include "utils.hpp"

// Finite-difference Schur recursion for hafnians and loop hafnians with repeated
// rows. The idea behind this algorithm will be described in a forthcoming
// arXiv preprint.
//
// Every level eliminates one edge of the matching of the photons by a 2x2 Schur
// complement over polynomials in t modulo t^(n+1), for every Glynn term of the
// edge, and accumulates the logarithms of the pivot factors in a phase. A leaf
// adds [t^n] exp(phase).

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
    if (eta == 0) {
        throw std::logic_error("edge of zero multiplicity");
    }
    const std::vector<long double> probabilities =
        binomialProbabilities(static_cast<std::int64_t>(eta));

    std::vector<GlynnTerm> terms;
    for (std::size_t k = 0; k <= eta; ++k) {
        const double z = static_cast<double>(eta) - 2.0 * static_cast<double>(k);
        if (root_symmetry && z < 0.0) {
            continue;
        }
        long double weight = (k % 2 ? -1.0L : 1.0L) * probabilities[k];
        if (root_symmetry && z > 0.0) {
            weight *= 2.0L;
        }
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
        largest = std::max(largest, std::norm(diagonal[row]));
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            largest = std::max(largest, std::abs(matrix(row, col)));
        }
    }
    if (largest > 0.0 && std::isfinite(largest)) {
        return static_cast<int>(std::lround(std::log2(largest) / 2));
    }
    return 0;
}

inline Complex rescale(const Complex &value, int exponent) {
    return {std::ldexp(value.real(), exponent), std::ldexp(value.imag(), exponent)};
}

struct Problem {
    std::size_t n = 0; // Truncation order, half the number of vertices.
    Matrix<Complex> matrix;
    Vector<Complex> mu;
    GlynnTerms glynn_terms; // For every edge of the matching, i.e., level.
    bool loops = true;
    int scale_exponent = 0;

    // Batch over 0, ..., outputs-1 copies of a marked vertex.
    std::size_t outputs = 1;
    Vector<Complex> marked;
    Complex marked_pair = 0.0;
    Complex marked_loop = 0.0;
};

inline bool is_batch(const Problem &problem) { return problem.outputs > 1; }

// The phase has components 1, x and x^2 in the weight x of the marked vertex.
inline std::size_t components(const Problem &problem) {
    return is_batch(problem) ? 3 : 1;
}

// Quadratic forms (mu, mu), (mu, marked) and (marked, marked) of the phase.
inline std::size_t forms(const Problem &problem) {
    return problem.loops ? components(problem) : 0;
}

constexpr double kFormFactors[3] = {0.5, 1.0, 0.5};

// The input scaled by `scale_exponent`, with rows and columns in the order of
// the edges of the matching. A dummy mode with one photon and loop weight one is
// prepended for an odd number of photons. The marked vertex of a batch is the
// last mode.
inline Problem make_problem(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal,
    const std::vector<std::int64_t> &occupations,
    bool loops,
    std::size_t outputs = 1
) {
    const std::int64_t photons =
        std::accumulate(occupations.begin(), occupations.end(), std::int64_t{0});
    const std::size_t pad = photons % 2 != 0 ? 1 : 0;
    std::vector<std::int64_t> padded(pad, 1);
    padded.insert(padded.end(), occupations.begin(), occupations.end());

    Problem problem;
    problem.n = (static_cast<std::size_t>(photons) + pad) / 2;
    problem.loops = loops;
    problem.outputs = outputs;
    problem.scale_exponent = scale_exponent(matrix, diagonal);

    const double pair_scale = std::ldexp(1.0, -2 * problem.scale_exponent);
    const double loop_scale = std::ldexp(1.0, -problem.scale_exponent);
    const auto pair_weight = [&](std::size_t i, std::size_t j) {
        return i < pad || j < pad ? Complex{}
                                  : pair_scale * matrix(i - pad, j - pad);
    };
    const auto loop_weight = [&](std::size_t i) {
        return loop_scale * (i < pad ? Complex(1.0) : diagonal[i - pad]);
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
        for (std::size_t j = 0; j < dim; ++j) {
            problem.matrix(i, j) = pair_weight(modes[i], modes[j]);
        }
    }

    if (is_batch(problem)) {
        const std::size_t last = matrix.rows - 1 + pad;
        problem.marked = Vector<Complex>(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            problem.marked[i] = pair_weight(modes[i], last);
        }
        problem.marked_pair = pair_weight(last, last);
        problem.marked_loop = loop_weight(last);
    }

    return problem;
}

// The state of a level: the remaining matrix A (a packed upper triangle), loop
// weights, marked column and accumulated phase, followed by the parts of the
// pivot shared by all Glynn terms.
struct Level {
    std::size_t dim = 0;
    PolynomialArray A;
    PolynomialArray mu;
    PolynomialArray marked;
    PolynomialArray phase;
    PolynomialArray S;
    PolynomialArray beta;
    PolynomialArray Y;
    PolynomialArray X;
};

inline std::size_t packed_index(std::size_t dim, std::size_t i, std::size_t j) {
    return i * (2 * dim - i + 1) / 2 + (j - i);
}

inline Polynomial entry(Level &state, std::size_t i, std::size_t j) {
    return state.A[packed_index(state.dim, i, j)];
}

inline ConstPolynomial entry(const Level &state, std::size_t i, std::size_t j) {
    return state.A[packed_index(state.dim, i, j)];
}

// The buffers of the depth-first recursion of one thread.
struct Workspace {
    std::vector<Level> levels;
    PolynomialArray det;
    PolynomialArray inverse;
    PolynomialArray log_factor;
    PolynomialArray scratch;
    PolynomialArray U;
    PolynomialArray leaf_phase;
    PolynomialArray R;
};

// Allocates the levels from `first_level` on. The root level, if allocated,
// holds the input.
inline Workspace make_workspace(const Problem &problem, std::size_t first_level) {
    const std::size_t n = problem.n;
    const std::size_t pairs = problem.glynn_terms.size();
    Workspace workspace;
    workspace.levels.resize(pairs);
    for (std::size_t level = first_level; level < pairs; ++level) {
        const std::size_t dim = 2 * (pairs - level);
        Level &state = workspace.levels[level];
        state.dim = dim;
        state.A = PolynomialArray(dim * (dim + 1) / 2, n);
        state.mu = PolynomialArray(problem.loops ? dim : 0, n);
        state.marked = PolynomialArray(is_batch(problem) ? dim : 0, n);
        state.phase = PolynomialArray(components(problem), n + 1);
        state.S = PolynomialArray(1, n);
        state.beta = PolynomialArray(forms(problem), n);
        state.Y = PolynomialArray(forms(problem), n);
        state.X = PolynomialArray(2 * (dim - 2), n);
    }
    const std::size_t rows = 2 * (pairs - first_level) - 2;
    workspace.det = PolynomialArray(1, n + 1);
    workspace.inverse = PolynomialArray(1, n);
    workspace.log_factor = PolynomialArray(components(problem), n + 1);
    workspace.scratch = PolynomialArray(2, n);
    workspace.U = PolynomialArray(2 * rows, n);
    workspace.leaf_phase = PolynomialArray(components(problem), n + 1);
    workspace.R = PolynomialArray(3, n + 1);

    if (first_level == 0) {
        Level &root = workspace.levels[0];
        for (std::size_t i = 0; i < root.dim; ++i) {
            for (std::size_t j = i; j < root.dim; ++j) {
                set_coefficient(entry(root, i, j), 0, problem.matrix(i, j));
            }
            if (problem.loops) {
                set_coefficient(root.mu[i], 0, problem.mu[i]);
            }
            if (is_batch(problem)) {
                set_coefficient(root.marked[i], 0, problem.marked[i]);
            }
        }
    }

    return workspace;
}

// With a = A00, b = A01 and c = A11: S = b^2 - ac, the forms
// x^T E y = (z beta + z^2 t Y) / det, and X, for which A21 E = (z (A_p1, A_p0)
// + z^2 t X) / det.
inline void prepare(
    const Problem &problem,
    Workspace &workspace,
    std::size_t level
) {
    Level &state = workspace.levels[level];
    const std::size_t n = problem.n;
    const ConstPolynomial a = entry(state, 0, 0);
    const ConstPolynomial b = entry(state, 0, 1);
    const ConstPolynomial c = entry(state, 1, 1);

    const Polynomial S = state.S[0];
    set_zero(S, n - 1);
    mul_add(S, b, b, n - 1);
    mul_add(S, a, c, n - 1, 0, -1.0);

    const Polynomial xy00 = workspace.scratch[0];
    const Polynomial xy11 = workspace.scratch[1];
    for (std::size_t form = 0; form < forms(problem); ++form) {
        const PolynomialArray &x = form == 2 ? state.marked : state.mu;
        const PolynomialArray &y = form == 0 ? state.mu : state.marked;
        const Polynomial beta = state.beta[form];
        set_zero(beta, n);
        mul_add(beta, x[0], y[1], n);
        mul_add(beta, x[1], y[0], n);
        set_zero(xy00, n - 1);
        mul_add(xy00, x[0], y[0], n - 1);
        set_zero(xy11, n - 1);
        mul_add(xy11, x[1], y[1], n - 1);

        const Polynomial Y = state.Y[form];
        set_zero(Y, n - 1);
        mul_add(Y, c, xy00, n - 1);
        mul_add(Y, a, xy11, n - 1);
        mul_add(Y, b, beta, n - 1, 0, -1.0);
    }

    for (std::size_t row = 0; row + 2 < state.dim; ++row) {
        const ConstPolynomial Ap0 = entry(state, 0, row + 2);
        const ConstPolynomial Ap1 = entry(state, 1, row + 2);
        const Polynomial X0 = state.X[2 * row];
        const Polynomial X1 = state.X[2 * row + 1];
        set_zero(X0, n - 2);
        mul_add(X0, Ap0, c, n - 2);
        mul_add(X0, Ap1, b, n - 2, 0, -1.0);
        set_zero(X1, n - 2);
        mul_add(X1, Ap1, a, n - 2);
        mul_add(X1, Ap0, b, n - 2, 0, -1.0);
    }
}

// destination = phase + log of the pivot factors of the signed count z, with
// the pivot E = z X_2 (I - t z A11 X_2)^-1, and workspace.inverse = 1 / det.
inline void pivot(
    const Problem &problem,
    Workspace &workspace,
    std::size_t level,
    double z,
    PolynomialArray &destination
) {
    const Level &state = workspace.levels[level];
    const std::size_t n = problem.n;
    if (z == 0.0) {
        for (std::size_t component = 0; component < components(problem);
             ++component) {
            copy(destination[component], state.phase[component], n + 1);
        }
        return;
    }

    // det(I - t z A11 X_2) = 1 - 2 z t b + z^2 t^2 S
    const Polynomial det = workspace.det[0];
    const Polynomial inverse = workspace.inverse[0];
    set_coefficient(det, 0, 1.0);
    combine({det.re + 1, det.im + 1}, -2.0 * z, entry(state, 0, 1), z * z,
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
        mul_add(log_factor[form], numerator, inverse, n + 1, 1, kFormFactors[form]);
    }

    for (std::size_t component = 0; component < components(problem); ++component) {
        add(destination[component], state.phase[component], log_factor[component],
            n + 1);
    }
}

// Eliminates the edge of `level` with the signed count z, giving
// A22 + t A21 E A12 for the next level, and similarly for the vectors.
inline void descend(
    const Problem &problem,
    Workspace &workspace,
    std::size_t level,
    double z
) {
    const Level &parent = workspace.levels[level];
    Level &child = workspace.levels[level + 1];
    const std::size_t n = problem.n;
    const std::size_t rows = parent.dim - 2;
    PolynomialArray &U = workspace.U;

    pivot(problem, workspace, level, z, child.phase);

    // U = A21 E
    if (z != 0.0) {
        const Polynomial numerator = workspace.scratch[0];
        const auto divide = [&](Polynomial out, ConstPolynomial x,
                                ConstPolynomial y) {
            combine(numerator, z, x, z * z, y, n - 1);
            set_zero(out, n - 1);
            mul_add(out, numerator, workspace.inverse[0], n - 1);
        };
        for (std::size_t row = 0; row < rows; ++row) {
            divide(U[2 * row], entry(parent, 1, row + 2), parent.X[2 * row]);
            divide(U[2 * row + 1], entry(parent, 0, row + 2),
                   parent.X[2 * row + 1]);
        }
    }

    // target = value + t U_i (x0, x1)^T
    const auto update = [&](Polynomial target, ConstPolynomial value,
                            std::size_t i, ConstPolynomial x0,
                            ConstPolynomial x1) {
        copy(target, value, n);
        if (z != 0.0) {
            mul_add(target, U[2 * i], x0, n, 1);
            mul_add(target, U[2 * i + 1], x1, n, 1);
        }
    };
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = i; j < rows; ++j) {
            update(entry(child, i, j), entry(parent, i + 2, j + 2), i,
                   entry(parent, 0, j + 2), entry(parent, 1, j + 2));
        }
        if (problem.loops) {
            update(child.mu[i], parent.mu[i + 2], i, parent.mu[0], parent.mu[1]);
        }
        if (is_batch(problem)) {
            update(child.marked[i], parent.marked[i + 2], i, parent.marked[0],
                   parent.marked[1]);
        }
    }
}

// Adds weight [t^n] R_k for every output k, where R_0 = exp(phase) and
// k R_k = beta R_(k-1) + 2 gamma R_(k-2) for the copies of the marked vertex.
inline void leaf(
    const Problem &problem,
    Workspace &workspace,
    std::size_t level,
    double z,
    double weight,
    CompensatedSum<Complex> *sums
) {
    const std::size_t n = problem.n;
    PolynomialArray &phase = workspace.leaf_phase;
    pivot(problem, workspace, level, z, phase);

    Polynomial previous2 = workspace.R[0];
    Polynomial previous = workspace.R[1];
    Polynomial current = workspace.R[2];
    exponential(previous, phase[0], n, workspace.scratch[0]);
    compensated_add(sums[0], weight * coefficient(previous, n));
    if (!is_batch(problem)) {
        return;
    }

    const Polynomial beta = phase[1];
    const Polynomial gamma = phase[2];
    set_coefficient(beta, 0, coefficient(beta, 0) + problem.marked_loop);
    set_coefficient(gamma, 0, coefficient(gamma, 0) + 0.5 * problem.marked_pair);
    for (std::size_t k = 1; k < problem.outputs; ++k) {
        const double inverse_k = 1.0 / static_cast<double>(k);
        set_zero(current, n + 1);
        mul_add(current, beta, previous, n + 1, 0, inverse_k);
        if (k >= 2) {
            mul_add(current, gamma, previous2, n + 1, 0, 2.0 * inverse_k);
        }
        compensated_add(sums[k], weight * coefficient(current, n));
        std::swap(previous2, previous);
        std::swap(previous, current);
    }
}

// Adds the leaves below the Glynn terms [first, past) of `level`.
inline void evaluate(
    const Problem &problem,
    Workspace &workspace,
    std::size_t level,
    std::size_t first,
    std::size_t past,
    long double weight,
    CompensatedSum<Complex> *sums
) {
    const std::vector<GlynnTerm> &terms = problem.glynn_terms[level];
    prepare(problem, workspace, level);
    for (std::size_t k = first; k < past; ++k) {
        const long double child_weight = weight * terms[k].weight;
        if (level + 1 == problem.glynn_terms.size()) {
            leaf(problem, workspace, level, terms[k].signed_count,
                 static_cast<double>(child_weight), sums);
        } else {
            descend(problem, workspace, level, terms[k].signed_count);
            evaluate(problem, workspace, level + 1, 0,
                     problem.glynn_terms[level + 1].size(), child_weight, sums);
        }
    }
}

// The Glynn terms [first_term, past_term) of a level, below a fixed path.
struct Task {
    std::size_t first_term = 0;
    std::size_t past_term = 0;
    long double weight = 1.0L;
    Level state;
};

// Expands the tree down to `cut`, splitting the Glynn terms there into `chunks`.
inline void collect(
    const Problem &problem,
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
    prepare(problem, workspace, level);
    for (const GlynnTerm &term : terms) {
        descend(problem, workspace, level, term.signed_count);
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
    const Problem &problem,
    Workspace &root,
    std::vector<CompensatedSum<Complex>> &sums
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
    std::vector<CompensatedSum<Complex>> partial(tasks.size() * outputs);
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
        evaluate(problem, workspace, cut, tasks[t].first_term,
                 tasks[t].past_term, tasks[t].weight, &partial[t * outputs]);
    }

    for (std::size_t t = 0; t < tasks.size(); ++t) {
        for (std::size_t k = 0; k < outputs; ++k) {
            compensated_add(sums[k], compensated_value(partial[t * outputs + k]));
        }
    }
}

// The loop hafnians with 0, ..., outputs-1 copies of the marked vertex, each
// divided by the factorial of the number of copies.
inline std::vector<Complex> evaluate_problem(const Problem &problem) {
    const std::size_t outputs = problem.outputs;

    long double work = static_cast<long double>(problem.n)
        * static_cast<long double>(problem.n) * static_cast<long double>(outputs);
    for (const std::vector<GlynnTerm> &terms : problem.glynn_terms) {
        work *= static_cast<long double>(terms.size());
    }

    Workspace root = make_workspace(problem, 0);
    std::vector<CompensatedSum<Complex>> sums(outputs);
    if (work < kParallelWorkThreshold) {
        evaluate(problem, root, 0, 0, problem.glynn_terms[0].size(), 1.0L,
                 sums.data());
    } else {
        evaluate_in_parallel(problem, root, sums);
    }

    std::vector<Complex> result(outputs);
    for (std::size_t k = 0; k < outputs; ++k) {
        result[k] = rescale(
            compensated_value(sums[k]),
            problem.scale_exponent * static_cast<int>(2 * problem.n + k)
        );
    }

    return result;
}

} // namespace piquasso::hafnian

#endif
