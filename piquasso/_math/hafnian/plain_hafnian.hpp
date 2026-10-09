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

#ifndef PIQUASSO_HAFNIAN_PLAIN_HAFNIAN_HPP
#define PIQUASSO_HAFNIAN_PLAIN_HAFNIAN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <utils.hpp>

#include "matrix.hpp"
#include "powtrace.hpp"
#include "utils.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T>
void calculate_reduced_matrix(
    const Matrix<T> &matrix,
    const std::vector<std::int64_t> &delta,
    Workspace<T> &workspace
) {
    std::vector<std::size_t> &nonzero_indices = workspace.nonzero_indices;
    nonzero_indices.clear();
    for (std::size_t index = 0; index < delta.size(); ++index) {
        if (delta[index] != 0) {
            nonzero_indices.push_back(index);
        }
    }

    Matrix<T> &reduced = workspace.reduced_matrix;
    reduced.rows = 2 * nonzero_indices.size();
    reduced.cols = reduced.rows;

    for (std::size_t index = 0; index < nonzero_indices.size(); ++index) {
        const std::size_t even = 2 * index;
        const std::size_t odd = even + 1;
        const T delta_value = static_cast<T>(
            static_cast<double>(delta[nonzero_indices[index]]));
        const std::size_t source_col_even = 2 * nonzero_indices[index] + 1;
        const std::size_t source_col_odd = 2 * nonzero_indices[index];

        for (std::size_t row = 0; row < reduced.rows; ++row) {
            const std::size_t source_row =
                2 * nonzero_indices[row / 2] + row % 2;
            reduced(row, even) =
                delta_value * matrix(source_row, source_col_even);
            reduced(row, odd) =
                delta_value * matrix(source_row, source_col_odd);
        }
    }
}

template <typename T>
double scale_reduced_matrix(
    Matrix<T> &matrix,
    std::size_t maximum_power
) {
    // LAPACK-style scaled sum of squares: unlike a direct sum of |a_ij|^2,
    // this remains finite for every finite matrix element.
    double maximum_absolute_value = 0.0;
    double scaled_squared_norm = 1.0;

    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            const double absolute_value = std::abs(matrix(row, col));
            if (absolute_value == 0.0) {
                continue;
            }

            if (maximum_absolute_value < absolute_value) {
                const double ratio = maximum_absolute_value / absolute_value;
                scaled_squared_norm = 1.0
                    + scaled_squared_norm * ratio * ratio;
                maximum_absolute_value = absolute_value;
            } else {
                const double ratio = absolute_value / maximum_absolute_value;
                scaled_squared_norm += ratio * ratio;
            }
        }
    }

    if (maximum_absolute_value == 0.0) {
        return 1.0;
    }

    const double logarithmic_norm = std::log(maximum_absolute_value)
        + 0.5 * std::log(scaled_squared_norm);
    const std::size_t effective_power = std::max(maximum_power, matrix.rows);
    const double safe_logarithmic_limit =
        std::log(std::numeric_limits<double>::max()) - std::log(16.0);

    // Avoid scaling unless matrix powers could approach overflow. Unnecessary
    // scaling magnifies rounding when calculate_f restores the original powers.
    if (
        logarithmic_norm <= 0.0
        || logarithmic_norm * static_cast<double>(effective_power)
            <= safe_logarithmic_limit
    ) {
        return 1.0;
    }

    const double target_logarithmic_norm = safe_logarithmic_limit
        / static_cast<double>(effective_power);
    const double scale_factor = std::exp(
        target_logarithmic_norm - logarithmic_norm
    );

    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            matrix(row, col) *= scale_factor;
        }
    }

    return scale_factor;
}

template <typename T>
const std::vector<T> &calculate_f(
    const std::vector<T> &traces,
    double scale_factor,
    Workspace<T> &workspace
) {
    const std::size_t half_dimension = traces.size();
    auto &auxiliary = workspace.f_auxiliary;
    auxiliary[0].assign(half_dimension + 1, T{});
    auxiliary[1].assign(half_dimension + 1, T{});
    auxiliary[0][0] = static_cast<T>(1.0);

    double inverse_scale_factor = 1.0 / scale_factor;
    std::size_t source = 0;
    std::size_t destination = 1;

    for (std::size_t index = 1; index <= half_dimension; ++index) {
        const T factor = traces[index - 1] * inverse_scale_factor
            / (2.0 * static_cast<double>(index));

        if (index % 2 == 1) {
            source = 0;
            destination = 1;
        } else {
            source = 1;
            destination = 0;
        }
        auxiliary[destination] = auxiliary[source];

        T power_factor = static_cast<T>(1.0);
        for (std::size_t multiple = 1;
             multiple <= half_dimension / index;
             ++multiple) {
            power_factor *= factor / static_cast<double>(multiple);
            const std::size_t start = index * multiple;
            for (std::size_t coefficient = start;
                 coefficient <= half_dimension;
                 ++coefficient) {
                auxiliary[destination][coefficient] +=
                    auxiliary[source][coefficient - start] * power_factor;
            }
        }

        inverse_scale_factor /= scale_factor;
    }

    return auxiliary[destination];
}

template <typename T>
T hafnian(
    const Matrix<T> &original_matrix,
    const std::vector<std::int64_t> &occupations
) {
    const std::int64_t particle_number = std::accumulate(
        occupations.begin(), occupations.end(), std::int64_t{0}
    );
    if (particle_number == 0) {
        return static_cast<T>(1.0);
    }
    if (particle_number % 2 != 0) {
        return T{};
    }

    const MatchedOccupations matched = match_occupation_numbers(occupations);
    Matrix<T> matrix = select_matrix(original_matrix, matched.edge_indices);
    const double scale_factor = primary_scale_factor(matrix);
    scale_matrix(matrix, scale_factor);

    const std::size_t half_dimension = static_cast<std::size_t>(std::accumulate(
        matched.edge_repetitions.begin(),
        matched.edge_repetitions.end(),
        std::int64_t{0}
    ));
    const std::size_t iteration_count =
        glynn_iteration_count(matched.edge_repetitions);
    // Exceptions cannot propagate safely out of an OpenMP region.
    for (const std::int64_t repetition : matched.edge_repetitions) {
        static_cast<void>(binomialCoeffDouble(repetition, repetition / 2));
    }

    const std::size_t job_count =
        glynn_parallel_job_count(iteration_count);
    const std::size_t workspace_count = glynn_workspace_count(job_count);
    std::vector<std::unique_ptr<Workspace<T>>> workspaces =
        create_workspaces<T>(
            workspace_count,
            matched.edge_repetitions.size(),
            matrix.rows,
            half_dimension
        );
    std::vector<T> partial_results(job_count, T{});
    std::vector<T> partial_compensations(job_count, T{});
    const std::int64_t omp_job_count =
        static_cast<std::int64_t>(job_count);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) if(glynn_parallel_enabled(job_count))
#endif
    for (std::int64_t omp_job_index = 0;
         omp_job_index < omp_job_count;
         ++omp_job_index) {
        const std::size_t job_index =
            static_cast<std::size_t>(omp_job_index);
        const GlynnJobRange range = glynn_job_range(
            iteration_count, job_count, job_index
        );
        T &partial_result = partial_results[job_index];
        T &partial_compensation = partial_compensations[job_index];
        Workspace<T> &workspace = *workspaces[glynn_worker_index()];

        for (std::size_t permutation = range.begin;
             permutation < range.end;
             ++permutation) {
            get_kept_edges(
                matched.edge_repetitions,
                permutation,
                workspace.kept_edges
            );
            const std::vector<std::int64_t> &kept_edges =
                workspace.kept_edges;
            bool negative = false;
            double combinatorial_factor = 1.0;
            std::vector<std::int64_t> &delta = workspace.delta;

            for (std::size_t edge = 0;
                 edge < matched.edge_repetitions.size();
                 ++edge) {
                const std::int64_t repetition =
                    matched.edge_repetitions[edge];
                const std::int64_t kept = kept_edges[edge];
                negative = negative != ((repetition - kept) % 2 != 0);
                combinatorial_factor *=
                    binomialCoeffDouble(repetition, kept);
                delta[edge] = 2 * kept - repetition;
            }

            calculate_reduced_matrix(matrix, delta, workspace);
            Matrix<T> &reduced = workspace.reduced_matrix;
            const double secondary_scale_factor =
                scale_reduced_matrix(reduced, half_dimension);
            calculate_power_traces(reduced, half_dimension, workspace);
            const std::vector<T> &coefficients = calculate_f(
                workspace.traces, secondary_scale_factor, workspace
            );
            const double prefactor =
                (negative ? -1.0 : 1.0) * combinatorial_factor;
            compensated_add(
                prefactor * coefficients[half_dimension],
                partial_result,
                partial_compensation
            );
        }
    }

    T result{};
    T result_compensation{};
    for (const T &partial_result : partial_results) {
        compensated_add(
            partial_result,
            result,
            result_compensation
        );
    }

    result *= std::pow(scale_factor, static_cast<double>(half_dimension));
    result /= std::ldexp(1.0, static_cast<int>(half_dimension) - 1);
    return result;
}

template <typename T>
std::vector<T> hafnian_batch(
    const Matrix<T> &original_matrix,
    const std::vector<std::int64_t> &occupations,
    std::int64_t cutoff
) {
    const std::int64_t particle_number = std::accumulate(
        occupations.begin(), occupations.end(), std::int64_t{0}
    );
    const std::int64_t odd = particle_number % 2;
    const std::int64_t result_size = odd != 0
        ? cutoff / 2 - 1
        : (cutoff - 1) / 2;
    std::vector<T> concatenated(static_cast<std::size_t>(cutoff), T{});
    // With no modes, the vacuum is the only nonzero batch entry.
    if (original_matrix.rows == 0) {
        if (!concatenated.empty()) {
            concatenated.front() = static_cast<T>(1.0);
        }
        return concatenated;
    }
    if (result_size < 0) {
        return concatenated;
    }

    std::vector<std::int64_t> adjusted_occupations = occupations;
    if (odd != 0) {
        ++adjusted_occupations.back();
    }

    MatchedOccupations matched = match_occupation_numbers(adjusted_occupations);
    matched.edge_repetitions.push_back(result_size);
    matched.edge_indices.push_back(original_matrix.rows - 1);
    matched.edge_indices.push_back(original_matrix.rows - 1);

    Matrix<T> matrix = select_matrix(original_matrix, matched.edge_indices);
    const double scale_factor = primary_scale_factor(matrix);
    scale_matrix(matrix, scale_factor);

    const std::int64_t half_dimension = std::accumulate(
        matched.edge_repetitions.begin(),
        matched.edge_repetitions.end(),
        std::int64_t{0}
    );
    const std::size_t iteration_count =
        glynn_iteration_count(matched.edge_repetitions);
    // Exceptions cannot propagate safely out of an OpenMP region.
    for (const std::int64_t repetition : matched.edge_repetitions) {
        static_cast<void>(binomialCoeffDouble(repetition, repetition / 2));
    }

    const std::size_t value_count =
        static_cast<std::size_t>(result_size + 1);
    const std::size_t job_count =
        glynn_parallel_job_count(iteration_count);
    const std::size_t workspace_count = glynn_workspace_count(job_count);
    std::vector<std::unique_ptr<Workspace<T>>> workspaces =
        create_workspaces<T>(
            workspace_count,
            matched.edge_repetitions.size(),
            matrix.rows,
            static_cast<std::size_t>(half_dimension)
        );
    std::vector<std::vector<T>> partial_results(
        job_count, std::vector<T>(value_count, T{})
    );
    std::vector<std::vector<T>> partial_compensations(
        job_count, std::vector<T>(value_count, T{})
    );
    const std::int64_t omp_job_count =
        static_cast<std::int64_t>(job_count);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) if(glynn_parallel_enabled(job_count))
#endif
    for (std::int64_t omp_job_index = 0;
         omp_job_index < omp_job_count;
         ++omp_job_index) {
        const std::size_t job_index =
            static_cast<std::size_t>(omp_job_index);
        const GlynnJobRange range = glynn_job_range(
            iteration_count, job_count, job_index
        );
        std::vector<T> &partial_result = partial_results[job_index];
        std::vector<T> &partial_compensation =
            partial_compensations[job_index];
        Workspace<T> &workspace = *workspaces[glynn_worker_index()];

        for (std::size_t permutation = range.begin;
             permutation < range.end;
             ++permutation) {
            get_kept_edges(
                matched.edge_repetitions,
                permutation,
                workspace.kept_edges
            );
            const std::vector<std::int64_t> &kept_edges =
                workspace.kept_edges;
            bool negative = result_size % 2 != 0;
            double combinatorial_factor = 1.0;
            std::vector<std::int64_t> &delta = workspace.delta;

            for (std::size_t edge = 0;
                 edge < matched.edge_repetitions.size();
                 ++edge) {
                const std::int64_t repetition =
                    matched.edge_repetitions[edge];
                const std::int64_t kept = kept_edges[edge];
                negative = negative != ((repetition - kept) % 2 != 0);
                if (edge + 1 != matched.edge_repetitions.size()) {
                    combinatorial_factor *=
                        binomialCoeffDouble(repetition, kept);
                }
                delta[edge] = 2 * kept - repetition;
            }

            calculate_reduced_matrix(matrix, delta, workspace);
            Matrix<T> &reduced = workspace.reduced_matrix;
            const double secondary_scale_factor = scale_reduced_matrix(
                reduced, static_cast<std::size_t>(half_dimension)
            );
            calculate_power_traces(
                reduced,
                static_cast<std::size_t>(half_dimension),
                workspace
            );
            const std::vector<T> &coefficients = calculate_f(
                workspace.traces, secondary_scale_factor, workspace
            );

            const std::int64_t kept_last = kept_edges.back();
            for (std::int64_t index = kept_last;
                 index <= result_size;
                 ++index) {
                const bool sign = negative != (index % 2 != 0);
                double prefactor =
                    (sign ? -1.0 : 1.0) * combinatorial_factor
                    * binomialCoeffDouble(index, kept_last);

                if (index >= result_size - kept_last) {
                    const double ratio = binomialCoeffDouble(
                        index, result_size - kept_last
                    ) / binomialCoeffDouble(index, kept_last);
                    prefactor *= 1.0
                        + (((result_size - index) % 2 != 0) ? -1.0 : 1.0)
                            * ratio;
                }

                const std::size_t result_index =
                    static_cast<std::size_t>(index);
                compensated_add(
                    prefactor * coefficients[static_cast<std::size_t>(
                        half_dimension - result_size + index
                    )],
                    partial_result[result_index],
                    partial_compensation[result_index]
                );
            }
        }
    }

    std::vector<T> result(value_count, T{});
    std::vector<T> result_compensation(value_count, T{});
    for (const std::vector<T> &partial_result : partial_results) {
        for (std::size_t index = 0; index < value_count; ++index) {
            compensated_add(
                partial_result[index],
                result[index],
                result_compensation[index]
            );
        }
    }

    for (std::int64_t index = 0; index <= result_size; ++index) {
        const std::int64_t exponent = half_dimension + index - result_size;
        result[static_cast<std::size_t>(index)] *=
            std::pow(scale_factor, static_cast<double>(exponent));
        result[static_cast<std::size_t>(index)] /=
            std::ldexp(1.0, static_cast<int>(exponent));
        concatenated[static_cast<std::size_t>(2 * index + odd)] =
            result[static_cast<std::size_t>(index)];
    }

    if (particle_number == 0) {
        concatenated[0] = static_cast<T>(1.0);
    }
    return concatenated;
}

} // namespace piquasso::hafnian

#endif
