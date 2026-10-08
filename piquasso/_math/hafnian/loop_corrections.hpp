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

#ifndef PIQUASSO_HAFNIAN_LOOP_CORRECTIONS_HPP
#define PIQUASSO_HAFNIAN_LOOP_CORRECTIONS_HPP

#include <cstddef>
#include <vector>

#include "matrix.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T>
void calculate_loop_corrections(
    const std::vector<T> &right_diagonal,
    const std::vector<T> &left_diagonal,
    const Matrix<T> &matrix,
    std::size_t number_of_modes,
    Workspace<T> &workspace
) {
    std::vector<T> &corrections = workspace.loop_corrections;
    std::vector<T> &right_state = workspace.loop_right_state;
    std::vector<T> &temporary = workspace.loop_temporary;
    corrections.assign(number_of_modes, T{});
    right_state.assign(right_diagonal.begin(), right_diagonal.end());
    temporary.resize(right_diagonal.size());

    if (right_state.empty()) {
        return;
    }

    for (std::size_t mode = 0; mode < number_of_modes; ++mode) {
        for (std::size_t index = 0; index < right_state.size(); ++index) {
            corrections[mode] += left_diagonal[index] * right_state[index];
        }

        T value{};
        for (std::size_t index = 0; index < right_state.size(); ++index) {
            value += matrix(0, index) * right_state[index];
        }
        temporary[0] = value;

        for (std::size_t row = 1; row < right_state.size(); ++row) {
            value = T{};
            for (std::size_t col = row - 1;
                 col < right_state.size();
                 ++col) {
                value += matrix(row, col) * right_state[col];
            }
            temporary[row] = value;
        }
        right_state.swap(temporary);
    }
}

} // namespace piquasso::hafnian

#endif
