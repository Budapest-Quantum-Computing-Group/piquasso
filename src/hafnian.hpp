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

#ifndef PIQUASSO_HAFNIAN_HPP
#define PIQUASSO_HAFNIAN_HPP

#include <complex>
#include <cstdint>
#include <vector>

#include "matrix.hpp"

// Loop hafnians of symmetric matrices with repeated rows and columns, by the
// parallel finite-difference Schur recursion. The idea behind this algorithm
// will be described in a forthcoming arXiv preprint.
//
// The batch adds k = 0, ..., cutoff-1 photons to the last mode.

namespace piquasso::hafnian {

std::complex<double> loop_hafnian(
    const Matrix<std::complex<double>> &matrix,
    const Vector<std::complex<double>> &diagonal,
    const std::vector<std::int64_t> &occupations
);

std::vector<std::complex<double>> loop_hafnian_batch(
    const Matrix<std::complex<double>> &matrix,
    const Vector<std::complex<double>> &diagonal,
    const std::vector<std::int64_t> &occupations,
    std::int64_t cutoff
);

} // namespace piquasso::hafnian

#endif
