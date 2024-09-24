/*
 * Copyright 2021-2024 Budapest Quantum Computing Group
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

#include "matrix.hpp"
#include "torontonian_common.hpp"

#include <complex>
#include <vector>
#include <cstring>

std::vector<size_t> calculate_reduce_indices(
    const std::vector<size_t> &selected_index_holes,
    const size_t &num_of_modes)
{
    size_t number_selected_modes = num_of_modes - selected_index_holes.size();

    std::vector<size_t> positions_of_ones;
    positions_of_ones.reserve(number_selected_modes);
    if (selected_index_holes.size() == 0)
        for (size_t idx = 0; idx < num_of_modes; idx++)
            positions_of_ones.push_back(idx);
    else
    {
        size_t hole_idx = 0;
        for (size_t idx = 0; idx < num_of_modes; idx++)
        {
            if (idx == (size_t)selected_index_holes[hole_idx] && hole_idx < selected_index_holes.size())
            {
                hole_idx++;
                continue;
            }
            positions_of_ones.push_back(idx);
        }
    }

    return positions_of_ones;
}

