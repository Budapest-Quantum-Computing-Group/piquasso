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

#ifndef TORONTONIAN_COMMON_H
#define TORONTONIAN_COMMON_H

#include "matrix.hpp"
#include <vector>
#include <complex>
#include <vector>
#include <cstring>

std::vector<size_t> calculate_reduce_indices(
    const std::vector<size_t> &selected_index_holes,
    const size_t &num_of_modes);


template <typename TScalar>
Matrix<TScalar> calc_id_minus_matrix(Matrix<TScalar> matrix_in) {
    size_t dim = matrix_in.rows;

    Matrix<TScalar> matrix(dim, dim);
    for (size_t idx = 0; idx < dim; idx++)
    {
        for (size_t jdx = 0; jdx < dim; jdx++)
            matrix[idx * matrix.stride + jdx] = -matrix_in[idx * matrix_in.stride + jdx];

        matrix[idx * dim + idx] += 1.0;
    }

    return matrix;
}


/**
 * @brief Copy data from the input matrix and the reusable partial Cholesky decomposition matrix L.
 */
template <typename TScalar>
void copy_after_reuse_index(
    Matrix<TScalar> &target,
    const Matrix<TScalar> &L,
    const Matrix<TScalar> &matrix,
    const std::vector<size_t> &positions_of_ones,
    size_t reuse_index
)
{
    size_t number_selected_modes = target.rows / 2;

    for (size_t idx = reuse_index; idx < number_selected_modes; idx++)
    {
        TScalar *matrix_data = matrix.data + 2 * (positions_of_ones[idx] * matrix.stride);
        TScalar *L_data = L.data + 2 * (idx + 1) * L.stride;
        TScalar *target_data = target.data + 2 * (idx * target.stride);

        memcpy(target_data, L_data, 2 * (idx + 1) * sizeof(TScalar));
        memcpy(target_data + target.stride, L_data + L.stride, 2 * (idx + 1) * sizeof(TScalar));

        for (size_t jdx = reuse_index; jdx <= idx; jdx++)
            memcpy(target_data + 2 * jdx, matrix_data + 2 * positions_of_ones[jdx], 2 * sizeof(TScalar));

        target_data = target_data + target.stride;
        matrix_data = matrix_data + matrix.stride;

        for (size_t jdx = reuse_index; jdx <= idx; jdx++)
            memcpy(target_data + 2 * jdx, matrix_data + 2 * positions_of_ones[jdx], 2 * sizeof(TScalar));
    }
}

template <typename TScalar>
void calc_cholesky_decomposition(
    Matrix<TScalar> &matrix,
    const size_t &reuse_index,
    TScalar &determinant)
{
    determinant = 1.0;

    // storing in the same memory the results of the algorithm
    size_t n = matrix.cols;
    // Decomposing a matrix into lower triangular matrices
    for (size_t i = reuse_index; i < n; i++)
    {
        TScalar *row_i = matrix.data + i * matrix.stride;

        for (size_t j = reuse_index; j < i; j++)
        {
            {
                TScalar *row_j = matrix.data + j * matrix.stride;

                TScalar sum = 0.0;

                for (size_t k = 0; k < j; k++)
                    sum += row_i[k] * row_j[k];

                row_i[j] = (row_i[j] - sum) / row_j[j];
            }
        }
        TScalar sum = 0.0;

        for (size_t k = 0; k < i; k++)
            sum += row_i[k] * row_i[k];

        row_i[i] = sqrt(row_i[i] - sum);
        determinant = determinant * row_i[i];
    }
}

template <typename TScalar>
void calc_determinant_cholesky_decomposition(
    Matrix<TScalar> &matrix,
    const size_t reuse_index,
    TScalar &determinant)
{
    determinant = 1.0;

    // calculate the rest of the Cholesky decomposition and calculate the determinant
    calc_cholesky_decomposition(matrix, reuse_index, determinant);

    // multiply the result with the remaining diagonal elements of the Cholesky matrix L,
    // that has been reused
    for (size_t idx = 0; idx < reuse_index; idx++)
        determinant *= matrix[idx * matrix.stride + idx];

    determinant = determinant * determinant;
}


template <typename TScalar>
TScalar calculate_partial_torontonian(
    const std::vector<size_t> &selected_index_holes,
    const TScalar &determinant,
    const size_t &num_of_modes)
{
    size_t number_selected_modes = num_of_modes - selected_index_holes.size();

    // calculating -1^(N-|Z|)
    TScalar factor =
        static_cast<TScalar>(
            (number_selected_modes + num_of_modes) % 2
                ? -1.0
                : 1.0);

    // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
    TScalar sqrt_determinant = std::sqrt(determinant);

    TScalar partial_torontonian = factor / sqrt_determinant;

    return partial_torontonian;
}

#endif