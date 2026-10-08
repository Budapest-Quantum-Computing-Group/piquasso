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

#ifndef PIQUASSO_HAFNIAN_UTILS_HPP
#define PIQUASSO_HAFNIAN_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "matrix.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T> T conjugate(const T &value) { return value; }

template <typename T>
std::complex<T> conjugate(const std::complex<T> &value) {
    return std::conj(value);
}

template <typename T> double squared_absolute_value(const T &value) {
    const double converted = static_cast<double>(value);
    return converted * converted;
}

template <typename T>
double squared_absolute_value(const std::complex<T> &value) {
    return static_cast<double>(std::norm(value));
}

template <typename T>
void compensated_add(const T &value, T &sum, T &compensation) {
    const T adjusted = value - compensation;
    const T updated = sum + adjusted;
    compensation = (updated - sum) - adjusted;
    sum = updated;
}

struct MatchedOccupations {
    std::vector<std::int64_t> edge_repetitions;
    std::vector<std::size_t> edge_indices;
};

struct GlynnJobRange {
    std::size_t begin;
    std::size_t end;
};

inline std::size_t glynn_parallel_job_count(std::size_t iteration_count) {
    constexpr std::size_t minimum_parallel_iteration_count = 64;
    if (iteration_count < minimum_parallel_iteration_count) {
        return iteration_count == 0 ? 0 : 1;
    }

    // Keep the partition independent of the OpenMP thread count so the final
    // merge, and therefore floating-point results, are reproducible.
    constexpr std::size_t maximum_job_count = 256;

    return std::min(iteration_count, maximum_job_count);
}

inline bool glynn_parallel_enabled(std::size_t job_count) {
#if defined(_OPENMP)
    return job_count > 1 && omp_get_max_threads() > 1;
#else
    static_cast<void>(job_count);
    return false;
#endif
}

inline std::size_t glynn_workspace_count(std::size_t job_count) {
    if (job_count == 0) {
        return 0;
    }

#if defined(_OPENMP)
    return std::min(
        job_count, static_cast<std::size_t>(omp_get_max_threads())
    );
#else
    return 1;
#endif
}

inline std::size_t glynn_worker_index() {
#if defined(_OPENMP)
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

template <typename T>
std::vector<std::unique_ptr<Workspace<T>>> create_workspaces(
    std::size_t workspace_count,
    std::size_t edge_count,
    std::size_t maximum_matrix_dimension,
    std::size_t maximum_power
) {
    std::vector<std::unique_ptr<Workspace<T>>> workspaces;
    workspaces.reserve(workspace_count);

    for (std::size_t index = 0; index < workspace_count; ++index) {
        workspaces.push_back(std::make_unique<Workspace<T>>(
            edge_count, maximum_matrix_dimension, maximum_power
        ));
    }

    return workspaces;
}

inline GlynnJobRange glynn_job_range(
    std::size_t iteration_count,
    std::size_t job_count,
    std::size_t job_index
) {
    const std::size_t common_size = iteration_count / job_count;
    const std::size_t remainder = iteration_count % job_count;
    const std::size_t begin = job_index * common_size
        + std::min(job_index, remainder);
    const std::size_t size = common_size + (job_index < remainder ? 1 : 0);

    return {begin, begin + size};
}

inline MatchedOccupations match_occupation_numbers(
    const std::vector<std::int64_t> &occupations
) {
    if (occupations.size() == 1) {
        return {{occupations[0] / 2}, {0, 0}};
    }

    std::vector<std::int64_t> remaining = occupations;
    MatchedOccupations result;

    while (std::accumulate(
        remaining.begin(), remaining.end(), std::int64_t{0}
    ) > 1) {
        std::vector<std::size_t> sorter(remaining.size());
        std::iota(sorter.begin(), sorter.end(), std::size_t{0});
        std::stable_sort(
            sorter.begin(), sorter.end(),
            [&remaining](std::size_t left, std::size_t right) {
                return remaining[left] < remaining[right];
            }
        );

        const std::size_t largest_index = sorter.back();
        const std::size_t second_largest_index = sorter[sorter.size() - 2];
        const std::int64_t largest_half = remaining[largest_index] / 2;
        const std::int64_t second_largest = remaining[second_largest_index];

        if (largest_half > second_largest) {
            remaining[largest_index] -= 2 * largest_half;
            result.edge_repetitions.push_back(largest_half);
            result.edge_indices.push_back(largest_index);
            result.edge_indices.push_back(largest_index);
        } else {
            remaining[largest_index] -= second_largest;
            remaining[second_largest_index] = 0;
            result.edge_repetitions.push_back(second_largest);
            result.edge_indices.push_back(largest_index);
            result.edge_indices.push_back(second_largest_index);
        }
    }

    return result;
}

template <typename T>
Matrix<T> select_matrix(
    const Matrix<T> &matrix,
    const std::vector<std::size_t> &indices
) {
    Matrix<T> selected(indices.size(), indices.size());

    for (std::size_t row = 0; row < indices.size(); ++row) {
        for (std::size_t col = 0; col < indices.size(); ++col) {
            selected(row, col) = matrix(indices[row], indices[col]);
        }
    }

    return selected;
}

template <typename T>
std::vector<T> select_vector(
    const std::vector<T> &vector,
    const std::vector<std::size_t> &indices
) {
    std::vector<T> selected(indices.size());

    for (std::size_t index = 0; index < indices.size(); ++index) {
        selected[index] = vector[indices[index]];
    }

    return selected;
}

inline void get_kept_edges(
    const std::vector<std::int64_t> &edge_repetitions,
    std::size_t index,
    std::vector<std::int64_t> &result
) {
    result.resize(edge_repetitions.size());

    for (std::size_t edge = 0; edge < edge_repetitions.size(); ++edge) {
        const std::size_t radix = static_cast<std::size_t>(
            edge_repetitions[edge] + 1
        );
        result[edge] = static_cast<std::int64_t>(index % radix);
        index /= radix;
    }
}

inline std::vector<std::int64_t> get_kept_edges(
    const std::vector<std::int64_t> &edge_repetitions,
    std::size_t index
) {
    std::vector<std::int64_t> result;
    get_kept_edges(edge_repetitions, index, result);

    return result;
}

inline std::size_t glynn_iteration_count(
    const std::vector<std::int64_t> &edge_repetitions
) {
    std::size_t count = 1;

    for (const std::int64_t repetition : edge_repetitions) {
        if (repetition < 0) {
            return 0;
        }

        const std::size_t factor = static_cast<std::size_t>(repetition + 1);
        if (
            factor != 0
            && count > std::numeric_limits<std::size_t>::max() / factor
        ) {
            throw std::overflow_error("The hafnian iteration count is too large.");
        }
        count *= factor;
    }

    return count / 2;
}

template <typename T>
double primary_scale_factor(const Matrix<T> &matrix) {
    if (matrix.rows <= 10) {
        return 1.0;
    }

    double maximum_absolute_value = 0.0;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            maximum_absolute_value = std::max(
                maximum_absolute_value,
                static_cast<double>(std::abs(matrix(row, col)))
            );
        }
    }

    if (maximum_absolute_value == 0.0) {
        return 1.0;
    }

    double scaled_absolute_sum = 0.0;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            scaled_absolute_sum +=
                std::abs(matrix(row, col)) / maximum_absolute_value;
        }
    }

    const double dimension = static_cast<double>(matrix.rows);
    const double scale_factor = maximum_absolute_value
        * (scaled_absolute_sum
            / (dimension * dimension * std::sqrt(2.0)));

    return scale_factor == 0.0 ? maximum_absolute_value : scale_factor;
}

template <typename T>
void scale_matrix(Matrix<T> &matrix, double scale_factor) {
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            matrix(row, col) /= scale_factor;
        }
    }
}

template <typename T>
void scale_matrix_and_diagonal(
    Matrix<T> &matrix,
    std::vector<T> &diagonal,
    double scale_factor
) {
    scale_matrix(matrix, scale_factor);
    const double diagonal_scale = std::sqrt(1.0 / scale_factor);
    for (T &value : diagonal) {
        value *= diagonal_scale;
    }
}

} // namespace piquasso::hafnian

#endif
