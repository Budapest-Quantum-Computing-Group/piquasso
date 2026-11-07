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

#ifndef PIQUASSO_MATCHING_HPP
#define PIQUASSO_MATCHING_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace piquasso::hafnian {

// The modes `first` and `second` are paired `multiplicity` times.
struct MatchingEdge {
    std::size_t first = 0;
    std::size_t second = 0;
    std::size_t multiplicity = 0;
};

using Matching = std::vector<MatchingEdge>;

// Greedy matching of the photons.
inline Matching match_occupation_numbers(
    const std::vector<std::int64_t> &occupations
) {
    if (occupations.size() == 1) {
        return {{0, 0, static_cast<std::size_t>(occupations[0] / 2)}};
    }

    std::vector<std::int64_t> remaining = occupations;
    Matching matching;
    while (std::accumulate(remaining.begin(), remaining.end(), std::int64_t{0})
           > 1) {
        std::vector<std::size_t> sorter(remaining.size());
        std::iota(sorter.begin(), sorter.end(), std::size_t{0});
        std::stable_sort(
            sorter.begin(), sorter.end(),
            [&remaining](std::size_t left, std::size_t right) {
                return remaining[left] < remaining[right];
            }
        );

        const std::size_t largest = sorter.back();
        const std::size_t second_largest = sorter[sorter.size() - 2];
        const std::int64_t half = remaining[largest] / 2;
        const std::int64_t second_count = remaining[second_largest];
        if (half > second_count) {
            matching.push_back({largest, largest, static_cast<std::size_t>(half)});
            remaining[largest] -= 2 * half;
        } else {
            matching.push_back(
                {largest, second_largest, static_cast<std::size_t>(second_count)}
            );
            remaining[largest] -= second_count;
            remaining[second_largest] = 0;
        }
    }

    return matching;
}

} // namespace piquasso::hafnian

#endif
