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

#ifndef PIQUASSO_HAFNIAN_POWTRACE_HPP
#define PIQUASSO_HAFNIAN_POWTRACE_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "hessenberg.hpp"
#include "labudde.hpp"
#include "matrix.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T>
void power_traces_from_charpoly(
    const Matrix<T> &coefficients,
    std::size_t maximum_power,
    std::vector<T> &traces
) {
    const std::size_t dimension = coefficients.rows;
    if (maximum_power == 0) {
        traces.assign(1, static_cast<T>(static_cast<double>(dimension)));
        return;
    }

    traces.assign(maximum_power, T{});
    if (dimension == 0) {
        return;
    }

    traces[0] = -coefficients(dimension - 1, 0);

    const std::size_t initial_maximum = std::min(maximum_power, dimension);
    for (std::size_t power = 2; power <= initial_maximum; ++power) {
        traces[power - 1] =
            -coefficients(dimension - 1, power - 1)
            * static_cast<double>(power);
        for (std::size_t offset = power - 1; offset > 0; --offset) {
            traces[power - 1] -=
                coefficients(dimension - 1, power - 1 - offset)
                * traces[offset - 1];
        }
    }

    for (std::size_t power = dimension; power < maximum_power; ++power) {
        traces[power] = T{};
        for (std::size_t offset = 1; offset <= dimension; ++offset) {
            traces[power] -=
                traces[power - offset]
                * coefficients(dimension - 1, offset - 1);
        }
    }
}

template <typename T>
void calculate_power_traces(
    Matrix<T> &matrix,
    std::size_t maximum_power,
    Workspace<T> &workspace
) {
    transform_to_hessenberg(matrix, workspace);
    labudde(
        matrix,
        workspace.characteristic_coefficients,
        workspace.beta_products
    );
    power_traces_from_charpoly(
        workspace.characteristic_coefficients,
        maximum_power,
        workspace.traces
    );
}

template <typename T>
void calculate_power_traces_loop(
    std::vector<T> &right_diagonal,
    std::vector<T> &left_diagonal,
    Matrix<T> &matrix,
    std::size_t maximum_power,
    Workspace<T> &workspace
) {
    transform_to_hessenberg(
        matrix, workspace, &left_diagonal, &right_diagonal
    );
    labudde(
        matrix,
        workspace.characteristic_coefficients,
        workspace.beta_products
    );
    power_traces_from_charpoly(
        workspace.characteristic_coefficients,
        maximum_power,
        workspace.traces
    );
}

} // namespace piquasso::hafnian

#endif
