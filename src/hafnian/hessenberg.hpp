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

#ifndef PIQUASSO_SRC_HAFNIAN_HESSENBERG_HPP
#define PIQUASSO_SRC_HAFNIAN_HESSENBERG_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "matrix.hpp"
#include "utils.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T>
bool reflection_vector(
    const Matrix<T> &matrix,
    std::size_t first_row,
    std::size_t column,
    std::vector<T> &reflection
) {
    reflection.resize(matrix.rows - first_row);
    double squared_norm = 0.0;

    for (std::size_t index = 0; index < reflection.size(); ++index) {
        reflection[index] = matrix(first_row + index, column);
        squared_norm += squared_absolute_value(reflection[index]);
    }

    const double sigma = std::sqrt(squared_norm);
    const double first_absolute = std::abs(reflection[0]);
    const double reflection_squared_norm =
        2.0 * (squared_norm + first_absolute * sigma);

    if (first_absolute != 0.0) {
        reflection[0] += reflection[0] / first_absolute * sigma;
    } else {
        reflection[0] += static_cast<T>(sigma);
    }

    if (reflection_squared_norm == 0.0) {
        return false;
    }

    const double reflection_norm = std::sqrt(reflection_squared_norm);
    for (T &element : reflection) {
        element /= reflection_norm;
    }

    return true;
}

template <typename T>
void apply_householder_rows(
    Matrix<T> &matrix,
    std::size_t first_row,
    std::size_t first_col,
    const std::vector<T> &reflection,
    const std::vector<T> &conjugated_reflection,
    std::vector<T> &reflected_row
) {
    const std::size_t column_count = matrix.cols - first_col;
    reflected_row.assign(column_count, T{});

    for (std::size_t index = 0; index < reflection.size(); ++index) {
        const T *matrix_row = matrix.data
            + (first_row + index) * matrix.stride + first_col;
        const T conjugated_coefficient = conjugated_reflection[index];

        for (std::size_t offset = 0; offset < column_count; ++offset) {
            reflected_row[offset] +=
                matrix_row[offset] * conjugated_coefficient;
        }
    }

    for (std::size_t index = 0; index < reflection.size(); ++index) {
        T *matrix_row = matrix.data
            + (first_row + index) * matrix.stride + first_col;
        const T coefficient = 2.0 * reflection[index];

        for (std::size_t offset = 0; offset < column_count; ++offset) {
            matrix_row[offset] -= coefficient * reflected_row[offset];
        }
    }
}

template <typename T>
void apply_householder_cols(
    Matrix<T> &matrix,
    std::size_t first_col,
    const std::vector<T> &reflection,
    const std::vector<T> &conjugated_reflection
) {
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        T *matrix_row = matrix.data + row * matrix.stride;
        T factor{};
        for (std::size_t index = 0; index < reflection.size(); ++index) {
            factor += matrix_row[first_col + index] * reflection[index];
        }
        factor *= 2.0;

        for (std::size_t index = 0; index < reflection.size(); ++index) {
            matrix_row[first_col + index] -=
                factor * conjugated_reflection[index];
        }
    }
}

template <typename T>
void transform_to_hessenberg(
    Matrix<T> &matrix,
    Workspace<T> &workspace,
    std::vector<T> *left_diagonal = nullptr,
    std::vector<T> *right_diagonal = nullptr
) {
    std::vector<T> &reflection = workspace.reflection;
    std::vector<T> &conjugated_reflection =
        workspace.conjugated_reflection;

    for (std::size_t index = 1; index + 1 < matrix.rows; ++index) {
        if (!reflection_vector(matrix, index, index - 1, reflection)) {
            continue;
        }

        conjugated_reflection.resize(reflection.size());
        std::transform(
            reflection.begin(),
            reflection.end(),
            conjugated_reflection.begin(),
            [](const T &value) { return conjugate(value); }
        );

        apply_householder_rows(
            matrix,
            index,
            index - 1,
            reflection,
            conjugated_reflection,
            workspace.reflected_row
        );

        if (left_diagonal != nullptr) {
            T factor{};
            for (std::size_t offset = 0; offset < reflection.size(); ++offset) {
                factor += (*left_diagonal)[index + offset] * reflection[offset];
            }
            factor *= 2.0;
            for (std::size_t offset = 0; offset < reflection.size(); ++offset) {
                (*left_diagonal)[index + offset] -=
                    factor * conjugated_reflection[offset];
            }
        }

        apply_householder_cols(
            matrix, index, reflection, conjugated_reflection
        );

        if (right_diagonal != nullptr) {
            T factor{};
            for (std::size_t offset = 0; offset < reflection.size(); ++offset) {
                factor += (*right_diagonal)[index + offset]
                    * conjugated_reflection[offset];
            }
            factor *= 2.0;
            for (std::size_t offset = 0; offset < reflection.size(); ++offset) {
                (*right_diagonal)[index + offset] -= reflection[offset] * factor;
            }
        }
    }
}

} // namespace piquasso::hafnian

#endif
