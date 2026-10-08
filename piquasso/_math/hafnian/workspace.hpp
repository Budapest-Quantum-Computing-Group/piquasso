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

#ifndef PIQUASSO_HAFNIAN_WORKSPACE_HPP
#define PIQUASSO_HAFNIAN_WORKSPACE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "matrix.hpp"

namespace piquasso::hafnian {

template <typename T> struct Workspace {
    std::vector<std::int64_t> kept_edges;
    std::vector<std::int64_t> delta;
    std::vector<std::size_t> nonzero_indices;

    Matrix<T> reduced_matrix;
    Matrix<T> characteristic_coefficients;
    std::vector<T> beta_products;
    std::vector<T> traces;
    std::array<std::vector<T>, 2> f_auxiliary;

    std::vector<T> left_diagonal;
    std::vector<T> right_diagonal;
    std::vector<T> loop_corrections;
    std::vector<T> loop_right_state;
    std::vector<T> loop_temporary;

    std::vector<T> reflection;
    std::vector<T> conjugated_reflection;
    std::vector<T> reflected_row;

    Workspace(
        std::size_t edge_count,
        std::size_t maximum_matrix_dimension,
        std::size_t maximum_power
    )
        : kept_edges(edge_count),
          delta(edge_count),
          reduced_matrix(
              maximum_matrix_dimension, maximum_matrix_dimension
          ),
          characteristic_coefficients(
              maximum_matrix_dimension, maximum_matrix_dimension
          ),
          beta_products(maximum_matrix_dimension),
          traces(maximum_power),
          f_auxiliary{
              std::vector<T>(maximum_power + 1),
              std::vector<T>(maximum_power + 1)
          },
          left_diagonal(maximum_matrix_dimension),
          right_diagonal(maximum_matrix_dimension),
          loop_corrections(maximum_power),
          loop_right_state(maximum_matrix_dimension),
          loop_temporary(maximum_matrix_dimension) {
        nonzero_indices.reserve(edge_count);
        reflection.reserve(maximum_matrix_dimension);
        conjugated_reflection.reserve(maximum_matrix_dimension);
        reflected_row.reserve(maximum_matrix_dimension);
    }
};

} // namespace piquasso::hafnian

#endif
