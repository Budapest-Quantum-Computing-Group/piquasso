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

#include "hafnian.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"
#include "schur_recursion.hpp"
#include "utils.hpp"

namespace piquasso::hafnian {

namespace {

// Signed counts are doubles and must remain exact integers.
constexpr std::int64_t kMaximumPhotonNumber = std::int64_t{1} << 40;

void validate_inputs(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal,
    const std::vector<std::int64_t> &occupations
) {
    if (matrix.rows != matrix.cols)
        throw std::invalid_argument("matrix must be square");

    if (diagonal.size() != matrix.rows || occupations.size() != matrix.rows) {
        throw std::invalid_argument(
            "diagonal and occupation numbers must match the matrix"
        );
    }

    std::int64_t photons = 0;
    for (const std::int64_t count : occupations) {
        if (count < 0)
            throw std::invalid_argument("occupation numbers must be nonnegative");

        if (count > kMaximumPhotonNumber - photons)
            throw std::invalid_argument("occupation numbers are too large");

        photons += count;
    }
}

} // namespace

// matrix(i,i) is the pair weight of distinct copies of mode i, and diagonal[i]
// is its loop weight. With a zero diagonal, this is the hafnian.
Complex loop_hafnian(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal,
    const std::vector<std::int64_t> &occupations
) {
    validate_inputs(matrix, diagonal, occupations);
    const std::int64_t photons =
        std::accumulate(occupations.begin(), occupations.end(), std::int64_t{0});
    const bool loops = !is_zero(diagonal);
    if (photons == 0)
        return 1.0;

    if (!loops && photons % 2 != 0)
        return 0.0;

    if (photons == 1) {
        for (std::size_t mode = 0; mode < matrix.rows; ++mode) {
            if (occupations[mode] != 0)
                return diagonal[mode];
        }
    }

    return evaluate_problem(make_problem(matrix, diagonal, occupations, loops))[0];
}

// The k additional photons of the last mode are copies of a marked vertex:
// sum_k L_k x^k / k! = exp(nu x + a x^2 / 2) lhaf(A, u + x w).
std::vector<Complex> loop_hafnian_batch(
    const Matrix<Complex> &matrix,
    const Vector<Complex> &diagonal,
    const std::vector<std::int64_t> &occupations,
    std::int64_t cutoff
) {
    validate_inputs(matrix, diagonal, occupations);
    if (cutoff < 1)
        throw std::invalid_argument("cutoff must be positive");

    const std::int64_t photons =
        std::accumulate(occupations.begin(), occupations.end(), std::int64_t{0});
    const auto outputs = static_cast<std::size_t>(cutoff);

    // L_k / k!
    std::vector<Complex> values(outputs, Complex{});
    if (matrix.rows == 0) {
        values[0] = 1.0;
    } else if (photons == 0) {
        const std::size_t last = matrix.rows - 1;
        values[0] = 1.0;
        for (std::size_t k = 1; k < outputs; ++k) {
            values[k] = diagonal[last] * values[k - 1];
            if (k >= 2)
                values[k] += matrix(last, last) * values[k - 2];

            values[k] /= static_cast<double>(k);
        }
    } else {
        values = evaluate_problem(
            make_problem(matrix, diagonal, occupations, true, outputs)
        );
    }

    // Without loops, an odd number of vertices has no perfect matching.
    if (is_zero(diagonal)) {
        for (std::size_t k = photons % 2 == 0 ? 1 : 0; k < outputs; k += 2)
            values[k] = 0.0;
    }

    double factorial = 1.0;
    for (std::size_t k = 0; k < outputs; ++k) {
        values[k] *= factorial;
        factorial *= static_cast<double>(k + 1);
    }

    return values;
}

} // namespace piquasso::hafnian
