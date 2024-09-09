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

/**
 * @brief Recursive torontonian implementation translated from PiquassoBoost.
 *
 * @note The torontonian is only implemented for reals, because one can always perform a
 * basis change on the input matrix so that it is real and the result is the same.
 *
 * See: https://arxiv.org/abs/2109.04528
 */

#include "matrix.hpp"
#include "torontonian.hpp"

#include <complex>
#include <vector>
#include <cstring>
#include <iostream>

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

    // multiply the result with the remaining diagonal elements of the Cholesky matrix L, that has been reused
    for (size_t idx = 0; idx < reuse_index; idx++)
        determinant *= matrix[idx * matrix.stride + idx];

    determinant = determinant * determinant;
}

template <typename TScalar>
Matrix<TScalar> reduce(
    const std::vector<size_t> &selected_index_holes,
    Matrix<TScalar> &L,
    const size_t &reuse_index,
    const Matrix<TScalar> &matrix,
    const size_t &num_of_modes)
{
    size_t number_selected_modes = num_of_modes - selected_index_holes.size();
    size_t dimension_of_AZ = 2 * number_selected_modes;
    Matrix<TScalar> AZ(dimension_of_AZ, dimension_of_AZ);

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

    // to calculate the determinant only the diagonal elements of L are necessary
    for (size_t idx = 0; idx < reuse_index; idx++)
    {
        AZ[2 * idx * AZ.stride + 2 * idx] = L[2 * idx * L.stride + 2 * idx];
        AZ[(2 * idx + 1) * AZ.stride + 2 * idx + 1] = L[(2 * idx + 1) * L.stride + 2 * idx + 1];
    }

    // copy data from the input matrix and the reusable partial Cholesky decomposition matrix L
    for (size_t idx = reuse_index; idx < number_selected_modes; idx++)
    {
        TScalar *matrix_data = matrix.data + 2 * (positions_of_ones[idx] * matrix.stride);
        TScalar *L_data = L.data + 2 * (idx + 1) * L.stride;
        TScalar *AZ_data = AZ.data + 2 * (idx * AZ.stride);

        memcpy(AZ_data, L_data, 2 * (idx + 1) * sizeof(TScalar));
        memcpy(AZ_data + AZ.stride, L_data + L.stride, 2 * (idx + 1) * sizeof(TScalar));

        for (size_t jdx = reuse_index; jdx <= idx; jdx++)
            memcpy(AZ_data + 2 * jdx, matrix_data + 2 * positions_of_ones[jdx], 2 * sizeof(TScalar));

        AZ_data = AZ_data + AZ.stride;
        matrix_data = matrix_data + matrix.stride;

        for (size_t jdx = reuse_index; jdx <= idx; jdx++)
            memcpy(AZ_data + 2 * jdx, matrix_data + 2 * positions_of_ones[jdx], 2 * sizeof(TScalar));
    }

    return AZ;
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

template <typename TScalar>
void iterate_over_selected_modes(
    std::vector<size_t> &selected_index_holes,
    size_t hole_to_iterate,
    Matrix<TScalar> &L,
    const size_t reuse_index,
    TScalar &sum,
    const size_t &num_of_modes,
    const Matrix<TScalar> &matrix)
{
    // calculate the partial Torontonian for the selected index holes
    size_t index_min;
    size_t index_max;
    if (hole_to_iterate == 0)
    {
        index_min = 0;
        index_max = selected_index_holes[hole_to_iterate] + 1;
    }
    else
    {
        index_min = selected_index_holes[hole_to_iterate - 1] + 1;
        index_max = selected_index_holes[hole_to_iterate] + 1;
    }

    // ***** iterations over the selected index hole to calculate partial torontonians *****

    // logical variable to control whether spawning new iterations or not
    bool stop_spawning_iterations = (selected_index_holes.size() == num_of_modes - 1);
    // add new index hole to the iterations
    size_t new_hole_to_iterate = hole_to_iterate + 1;

    // now do the rest of the iterations
    for (size_t idx = index_min + 1; idx != index_max; idx++)
    {
        std::vector<size_t> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[hole_to_iterate] = idx - 1;

        size_t reuse_index_new = std::min(idx - 1 - hole_to_iterate, reuse_index);

        Matrix<TScalar> &&L_new = reduce(
            selected_index_holes_new,
            L,
            reuse_index_new,
            matrix, num_of_modes);
        TScalar determinant;
        calc_determinant_cholesky_decomposition(L_new, 2 * reuse_index_new, determinant);

        TScalar partial_torontonian = calculate_partial_torontonian(selected_index_holes_new, determinant, num_of_modes);
        sum += partial_torontonian;

        // return if new index hole would give no nontrivial result
        // (in this case the partial torontonian is unity and should be counted only once in function torontonian_cpp)
        if (stop_spawning_iterations)
            continue;

        selected_index_holes_new.push_back(num_of_modes - 1);
        reuse_index_new = L_new.rows / 2 - 1;
        iterate_over_selected_modes(selected_index_holes_new, new_hole_to_iterate, L_new, reuse_index_new, sum, num_of_modes, matrix);
    }

    selected_index_holes[hole_to_iterate] = index_max - 1;

    size_t reuse_index_new = std::min(index_max - 1 - hole_to_iterate, reuse_index);

    Matrix<TScalar> &&L_new = reduce(selected_index_holes, L, reuse_index_new, matrix, num_of_modes);
    TScalar determinant;
    calc_determinant_cholesky_decomposition(L_new, 2 * reuse_index_new, determinant);

    TScalar partial_torontonian = calculate_partial_torontonian(
        selected_index_holes,
        determinant,
        num_of_modes);

    sum += partial_torontonian;
}

template <typename TScalar>
Matrix<TScalar> reorder_matrix(Matrix<TScalar> matrix)
{
    size_t dim = matrix.rows;

    size_t dim_over_2 = dim / 2;

    Matrix<TScalar> reordered_matrix(dim, dim);

    for (size_t idx = 0; idx < dim_over_2; idx++)
    {
        for (size_t jdx = 0; jdx < dim_over_2; jdx++)
        {
            reordered_matrix[2 * idx * reordered_matrix.stride + 2 * jdx] = matrix[idx * matrix.stride + jdx];
            reordered_matrix[2 * idx * reordered_matrix.stride + 2 * jdx + 1] = matrix[idx * matrix.stride + jdx + dim_over_2];
            reordered_matrix[(2 * idx + 1) * reordered_matrix.stride + 2 * jdx] = matrix[(idx + dim_over_2) * matrix.stride + jdx];
            reordered_matrix[(2 * idx + 1) * reordered_matrix.stride + 2 * jdx + 1] = matrix[(idx + dim_over_2) * matrix.stride + jdx + dim_over_2];
        }
    }

    return reordered_matrix;
}

template <typename TScalar>
TScalar torontonian_cpp(Matrix<TScalar> &matrix_in)
{
    Matrix<TScalar> matrix(matrix_in.rows, matrix_in.cols);
    size_t dim = matrix_in.rows;
    size_t num_of_modes = dim / 2;

    if (num_of_modes == 0)
        return 1.0;

    for (size_t idx = 0; idx < dim; idx++)
    {
        for (size_t jdx = 0; jdx < dim; jdx++)
            matrix[idx * matrix.stride + jdx] = -matrix_in[idx * matrix_in.stride + jdx];

        matrix[idx * dim + idx] += 1.0;
    }

    if (num_of_modes == 1)
    {
        TScalar determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        return (TScalar)1.0 / std::sqrt(determinant) - (TScalar)1.0;
    }

    Matrix<TScalar> &&reordered_matrix = reorder_matrix(matrix.copy());
    TScalar sum = 0.0;

    // construct the initial selection of the modes
    std::vector<size_t> selected_index_holes;

    // calculate the Cholesky decomposition of the initial matrix to be later reused
    Matrix<TScalar> L = reordered_matrix.copy();
    TScalar determinant;
    calc_determinant_cholesky_decomposition(L, 0, determinant);

    TScalar partial_torontonian = calculate_partial_torontonian(
        selected_index_holes,
        determinant,
        num_of_modes);

    sum += partial_torontonian;

    // add the first index hole in prior to the iterations
    selected_index_holes.push_back(num_of_modes - 1);

    // start task iterations originating from the initial selected modes
    iterate_over_selected_modes(selected_index_holes, 0, L, num_of_modes - 1, sum, num_of_modes, reordered_matrix);

    // last correction coming from an empty submatrix contribution
    TScalar factor = static_cast<TScalar>((num_of_modes) % 2 ? -1.0 : 1.0);
    sum = sum + factor;

    return sum;
}

template float torontonian_cpp<float>(Matrix<float> &matrix);
template double torontonian_cpp<double>(Matrix<double> &matrix);
