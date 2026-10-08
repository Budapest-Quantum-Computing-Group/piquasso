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

#ifndef PIQUASSO_HAFNIAN_LOOP_HAFNIAN_HPP
#define PIQUASSO_HAFNIAN_LOOP_HAFNIAN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <utils.hpp>

#include "loop_corrections.hpp"
#include "matrix.hpp"
#include "powtrace.hpp"
#include "utils.hpp"
#include "workspace.hpp"

namespace piquasso::hafnian {

template <typename T>
const std::vector<T> &calculate_f_loop(
    const std::vector<T> &traces,
    const std::vector<T> &loop_corrections,
    Workspace<T> &workspace
) {
    const std::size_t half_dimension = traces.size();
    auto &auxiliary = workspace.f_auxiliary;
    auxiliary[0].assign(half_dimension + 1, T{});
    auxiliary[1].assign(half_dimension + 1, T{});
    auxiliary[0][0] = static_cast<T>(1.0);

    std::size_t source = 0;
    std::size_t destination = 1;

    for (std::size_t index = 1; index <= half_dimension; ++index) {
        const T factor = traces[index - 1] / (2.0 * static_cast<double>(index))
            + loop_corrections[index - 1] * 0.5;

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
    }

    return auxiliary[destination];
}

template <typename T>
void calculate_loop_reduction(
    const Matrix<T> &matrix,
    const std::vector<T> &diagonal,
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

    const std::size_t dimension = 2 * nonzero_indices.size();
    Matrix<T> &reduced = workspace.reduced_matrix;
    std::vector<T> &left_diagonal = workspace.left_diagonal;
    std::vector<T> &right_diagonal = workspace.right_diagonal;
    reduced.rows = dimension;
    reduced.cols = dimension;
    left_diagonal.resize(dimension);
    right_diagonal.resize(dimension);

    for (std::size_t index = 0; index < nonzero_indices.size(); ++index) {
        const std::size_t even = 2 * index;
        const std::size_t odd = even + 1;
        const T delta_value = static_cast<T>(delta[nonzero_indices[index]]);
        const std::size_t source_col_even = 2 * nonzero_indices[index] + 1;
        const std::size_t source_col_odd = 2 * nonzero_indices[index];

        for (std::size_t row = 0; row < dimension; ++row) {
            const std::size_t source_row =
                2 * nonzero_indices[row / 2] + row % 2;
            reduced(row, even) =
                delta_value * matrix(source_row, source_col_even);
            reduced(row, odd) =
                delta_value * matrix(source_row, source_col_odd);
        }

        right_diagonal[even] = diagonal[source_col_odd];
        right_diagonal[odd] = diagonal[source_col_even];
        left_diagonal[even] = delta_value * diagonal[source_col_even];
        left_diagonal[odd] = delta_value * diagonal[source_col_odd];
    }
}

template <typename T>
const std::vector<T> &calculate_f_loop_from_matrix(
    const Matrix<T> &matrix,
    const std::vector<T> &diagonal,
    const std::vector<std::int64_t> &delta,
    std::size_t half_dimension,
    Workspace<T> &workspace
) {
    calculate_loop_reduction(matrix, diagonal, delta, workspace);
    calculate_power_traces_loop(
        workspace.right_diagonal,
        workspace.left_diagonal,
        workspace.reduced_matrix,
        half_dimension,
        workspace
    );
    calculate_loop_corrections(
        workspace.right_diagonal,
        workspace.left_diagonal,
        workspace.reduced_matrix,
        half_dimension,
        workspace
    );
    return calculate_f_loop(
        workspace.traces, workspace.loop_corrections, workspace
    );
}

template <typename T>
std::tuple<Matrix<T>, std::vector<T>, std::vector<std::int64_t>>
extend_loop_input(
    const Matrix<T> &matrix,
    const std::vector<T> &diagonal,
    const std::vector<std::int64_t> &occupations
) {
    Matrix<T> extended_matrix(matrix.rows + 1, matrix.cols + 1);
    for (std::size_t index = 0; index < extended_matrix.rows; ++index) {
        extended_matrix(index, 0) = T{};
        extended_matrix(0, index) = T{};
    }
    extended_matrix(0, 0) = static_cast<T>(1.0);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            extended_matrix(row + 1, col + 1) = matrix(row, col);
        }
    }

    std::vector<T> extended_diagonal(diagonal.size() + 1);
    extended_diagonal[0] = static_cast<T>(1.0);
    std::copy(diagonal.begin(), diagonal.end(), extended_diagonal.begin() + 1);

    std::vector<std::int64_t> extended_occupations(occupations.size() + 1);
    extended_occupations[0] = 1;
    std::copy(
        occupations.begin(), occupations.end(), extended_occupations.begin() + 1
    );

    return {
        std::move(extended_matrix),
        std::move(extended_diagonal),
        std::move(extended_occupations)
    };
}

template <typename T>
T loop_hafnian(
    const Matrix<T> &original_matrix,
    const std::vector<T> &original_diagonal,
    const std::vector<std::int64_t> &original_occupations
) {
    const std::int64_t particle_number = std::accumulate(
        original_occupations.begin(),
        original_occupations.end(),
        std::int64_t{0}
    );
    if (particle_number == 0) {
        return static_cast<T>(1.0);
    }

    Matrix<T> input_matrix;
    std::vector<T> input_diagonal;
    std::vector<std::int64_t> occupations;
    if (particle_number % 2 != 0) {
        std::tie(input_matrix, input_diagonal, occupations) =
            extend_loop_input(
                original_matrix, original_diagonal, original_occupations
            );
    } else {
        input_matrix = original_matrix.copy();
        input_diagonal = original_diagonal;
        occupations = original_occupations;
    }

    const MatchedOccupations matched = match_occupation_numbers(occupations);
    Matrix<T> matrix = select_matrix(input_matrix, matched.edge_indices);
    std::vector<T> diagonal = select_vector(input_diagonal, matched.edge_indices);
    const double scale_factor = primary_scale_factor(matrix);
    scale_matrix_and_diagonal(matrix, diagonal, scale_factor);

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

            const std::vector<T> &coefficients =
                calculate_f_loop_from_matrix(
                    matrix,
                    diagonal,
                    delta,
                    half_dimension,
                    workspace
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

struct AuxiliaryData {
    bool negative;
    double combinatorial_factor;
};

inline AuxiliaryData calculate_auxiliary_data(
    const std::vector<std::int64_t> &edge_repetitions,
    const std::vector<std::int64_t> &kept_edges,
    std::vector<std::int64_t> &delta
) {
    AuxiliaryData result{false, 1.0};
    delta.resize(edge_repetitions.size());

    for (std::size_t edge = 0; edge < edge_repetitions.size(); ++edge) {
        const std::int64_t repetition = edge_repetitions[edge];
        const std::int64_t kept = kept_edges[edge];
        result.negative = result.negative != ((repetition - kept) % 2 != 0);
        if (edge + 1 != edge_repetitions.size()) {
            result.combinatorial_factor *=
                binomialCoeffDouble(repetition, kept);
        }
        delta[edge] = 2 * kept - repetition;
    }

    return result;
}

template <typename T>
void accumulate_batch_summand(
    const std::vector<T> &coefficients,
    std::int64_t result_size,
    const std::vector<std::int64_t> &kept_edges,
    bool negative,
    double combinatorial_factor,
    std::vector<T> &result,
    std::vector<T> &compensation
) {
    const std::int64_t half_dimension =
        static_cast<std::int64_t>(coefficients.size()) - 1;
    const std::int64_t kept_last = kept_edges.back();

    for (std::int64_t index = kept_last; index <= result_size; ++index) {
        const bool sign = negative != ((index - result_size) % 2 != 0);
        double prefactor = (sign ? -1.0 : 1.0) * combinatorial_factor
            * binomialCoeffDouble(index, kept_last);

        if (index >= result_size - kept_last) {
            const double ratio = binomialCoeffDouble(
                index, result_size - kept_last
            ) / binomialCoeffDouble(index, kept_last);
            prefactor *= 1.0
                + (((result_size - index) % 2 != 0) ? -1.0 : 1.0) * ratio;
        }

        const std::size_t result_index = static_cast<std::size_t>(index);
        compensated_add(
            prefactor * coefficients[static_cast<std::size_t>(
                half_dimension - result_size + index
            )],
            result[result_index],
            compensation[result_index]
        );
    }
}

template <typename T> struct LoopBatchData {
    Matrix<T> matrix;
    std::vector<T> diagonal;
    Matrix<T> odd_matrix;
    std::vector<T> odd_diagonal;
    std::vector<std::int64_t> edge_repetitions;
    std::vector<std::int64_t> odd_edge_repetitions;
};

template <typename T>
LoopBatchData<T> prepare_loop_batch_data(
    const Matrix<T> &original_matrix,
    const std::vector<T> &original_diagonal,
    const std::vector<std::int64_t> &original_occupations,
    std::int64_t cutoff
) {
    const std::int64_t particle_number = std::accumulate(
        original_occupations.begin(),
        original_occupations.end(),
        std::int64_t{0}
    );
    const std::int64_t result_size = (cutoff - 1) / 2;
    const std::int64_t odd_result_size = cutoff >= 2 ? (cutoff - 2) / 2 : -1;
    std::vector<std::int64_t> occupations_copy = original_occupations;

    Matrix<T> matrix;
    std::vector<T> diagonal;
    Matrix<T> odd_matrix;
    std::vector<T> odd_diagonal;
    MatchedOccupations matched;
    MatchedOccupations odd_matched;

    if (particle_number % 2 == 0) {
        matched = match_occupation_numbers(original_occupations);
        matrix = original_matrix.copy();
        diagonal = original_diagonal;

        ++occupations_copy.back();
        std::vector<std::int64_t> unused_occupations;
        std::tie(odd_matrix, odd_diagonal, unused_occupations) =
            extend_loop_input(
                original_matrix, original_diagonal, occupations_copy
            );

        odd_matched.edge_repetitions.push_back(1);
        odd_matched.edge_repetitions.insert(
            odd_matched.edge_repetitions.end(),
            matched.edge_repetitions.begin(),
            matched.edge_repetitions.end()
        );
        odd_matched.edge_indices = {0, odd_matrix.rows - 1};
        for (const std::size_t index : matched.edge_indices) {
            odd_matched.edge_indices.push_back(index + 1);
        }
    } else {
        std::vector<std::int64_t> extended_occupations;
        std::tie(matrix, diagonal, extended_occupations) =
            extend_loop_input(
                original_matrix, original_diagonal, original_occupations
            );
        matched = match_occupation_numbers(extended_occupations);

        ++occupations_copy.back();
        odd_matrix = original_matrix.copy();
        odd_diagonal = original_diagonal;
        odd_matched = match_occupation_numbers(occupations_copy);
    }

    matched.edge_repetitions.push_back(result_size);
    matched.edge_indices.push_back(matrix.rows - 1);
    matched.edge_indices.push_back(matrix.rows - 1);

    odd_matched.edge_repetitions.push_back(odd_result_size);
    odd_matched.edge_indices.push_back(odd_matrix.rows - 1);
    odd_matched.edge_indices.push_back(odd_matrix.rows - 1);

    return {
        select_matrix(matrix, matched.edge_indices),
        select_vector(diagonal, matched.edge_indices),
        select_matrix(odd_matrix, odd_matched.edge_indices),
        select_vector(odd_diagonal, odd_matched.edge_indices),
        std::move(matched.edge_repetitions),
        std::move(odd_matched.edge_repetitions)
    };
}

template <typename T>
void accumulate_loop_batch(
    const Matrix<T> &matrix,
    const std::vector<T> &diagonal,
    const std::vector<std::int64_t> &edge_repetitions,
    std::int64_t result_size,
    std::vector<T> &result
) {
    if (result_size < 0) {
        return;
    }

    const std::size_t half_dimension = static_cast<std::size_t>(std::accumulate(
        edge_repetitions.begin(), edge_repetitions.end(), std::int64_t{0}
    ));
    const std::size_t iteration_count =
        glynn_iteration_count(edge_repetitions);
    // Exceptions cannot propagate safely out of an OpenMP region.
    for (const std::int64_t repetition : edge_repetitions) {
        static_cast<void>(binomialCoeffDouble(repetition, repetition / 2));
    }

    const std::size_t job_count =
        glynn_parallel_job_count(iteration_count);
    const std::size_t workspace_count = glynn_workspace_count(job_count);
    std::vector<std::unique_ptr<Workspace<T>>> workspaces =
        create_workspaces<T>(
            workspace_count,
            edge_repetitions.size(),
            matrix.rows,
            half_dimension
        );
    std::vector<std::vector<T>> partial_results(
        job_count, std::vector<T>(result.size(), T{})
    );
    std::vector<std::vector<T>> partial_compensations(
        job_count, std::vector<T>(result.size(), T{})
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
                edge_repetitions, permutation, workspace.kept_edges
            );
            const std::vector<std::int64_t> &kept_edges =
                workspace.kept_edges;
            const AuxiliaryData auxiliary = calculate_auxiliary_data(
                edge_repetitions, kept_edges, workspace.delta
            );
            const std::vector<T> &coefficients =
                calculate_f_loop_from_matrix(
                    matrix,
                    diagonal,
                    workspace.delta,
                    half_dimension,
                    workspace
                );
            accumulate_batch_summand(
                coefficients,
                result_size,
                kept_edges,
                auxiliary.negative,
                auxiliary.combinatorial_factor,
                partial_result,
                partial_compensation
            );
        }
    }

    std::vector<T> compensation(result.size(), T{});
    for (const std::vector<T> &partial_result : partial_results) {
        for (std::size_t index = 0; index < result.size(); ++index) {
            compensated_add(
                partial_result[index], result[index], compensation[index]
            );
        }
    }
}

template <typename T>
std::vector<T> loop_hafnian_batch(
    const Matrix<T> &original_matrix,
    const std::vector<T> &original_diagonal,
    const std::vector<std::int64_t> &original_occupations,
    std::int64_t cutoff
) {
    // With no modes, the vacuum is the only nonzero batch entry.
    if (original_matrix.rows == 0) {
        std::vector<T> result(static_cast<std::size_t>(cutoff), T{});
        if (!result.empty()) {
            result.front() = static_cast<T>(1.0);
        }
        return result;
    }

    const std::int64_t particle_number = std::accumulate(
        original_occupations.begin(),
        original_occupations.end(),
        std::int64_t{0}
    );
    LoopBatchData<T> data = prepare_loop_batch_data(
        original_matrix, original_diagonal, original_occupations, cutoff
    );

    const double scale_factor = primary_scale_factor(data.matrix);
    const double odd_scale_factor = primary_scale_factor(data.odd_matrix);
    scale_matrix_and_diagonal(data.matrix, data.diagonal, scale_factor);
    scale_matrix_and_diagonal(
        data.odd_matrix, data.odd_diagonal, odd_scale_factor
    );

    const std::int64_t result_size = (cutoff - 1) / 2;
    const std::int64_t odd_result_size = cutoff >= 2 ? (cutoff - 2) / 2 : -1;
    std::vector<T> even_result(
        static_cast<std::size_t>(result_size + 1), T{}
    );
    std::vector<T> odd_result(
        static_cast<std::size_t>(odd_result_size + 1), T{}
    );

    accumulate_loop_batch(
        data.matrix,
        data.diagonal,
        data.edge_repetitions,
        result_size,
        even_result
    );
    accumulate_loop_batch(
        data.odd_matrix,
        data.odd_diagonal,
        data.odd_edge_repetitions,
        odd_result_size,
        odd_result
    );

    const std::int64_t half_dimension = std::accumulate(
        data.edge_repetitions.begin(),
        data.edge_repetitions.end(),
        std::int64_t{0}
    );
    const std::int64_t odd_half_dimension = std::accumulate(
        data.odd_edge_repetitions.begin(),
        data.odd_edge_repetitions.end(),
        std::int64_t{0}
    );
    std::vector<T> result(static_cast<std::size_t>(cutoff), T{});

    for (std::size_t index = 0; index < even_result.size(); ++index) {
        const std::int64_t exponent = half_dimension
            - static_cast<std::int64_t>(even_result.size()) + 1
            + static_cast<std::int64_t>(index);
        result[2 * index] = even_result[index]
            / std::ldexp(1.0, static_cast<int>(exponent))
            * std::pow(scale_factor, static_cast<double>(exponent));
    }

    for (std::size_t index = 0; index < odd_result.size(); ++index) {
        const std::int64_t exponent = odd_half_dimension
            - static_cast<std::int64_t>(odd_result.size()) + 1
            + static_cast<std::int64_t>(index);
        result[2 * index + 1] = odd_result[index]
            / std::ldexp(1.0, static_cast<int>(exponent))
            * std::pow(odd_scale_factor, static_cast<double>(exponent));
    }

    if (particle_number == 0) {
        result[0] = static_cast<T>(1.0);
    }
    return result;
}

} // namespace piquasso::hafnian

#endif
