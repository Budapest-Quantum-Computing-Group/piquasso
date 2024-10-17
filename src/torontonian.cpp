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
#include "torontonian_common.hpp"

#include <complex>
#include <vector>
#include <cstring>
#include <iostream>


template <typename TScalar>
void copy_diagonals_partially(Matrix<TScalar> &target, const Matrix<TScalar> &source, size_t end)
{
    for (size_t idx = 0; idx < end; idx++)
        target[idx * target.stride + idx] = source[idx * source.stride + idx];
}


/**
 * Performs the reduction according to Fig. 2 from https://arxiv.org/abs/2109.04528,
 * but it only copies the diagonals until `reuse_index`.
 */
template <typename TScalar>
Matrix<TScalar> reduce_with_reuse(
    const std::vector<size_t> &positions_of_ones,
    Matrix<TScalar> &L,
    const size_t &reuse_index,
    const Matrix<TScalar> &matrix)
{
    auto dimension_of_AZ = 2 * positions_of_ones.size();

    Matrix<TScalar> AZ(dimension_of_AZ, dimension_of_AZ);

    copy_diagonals_partially(AZ, L, 2 * reuse_index);

    copy_after_reuse_index(AZ, L, matrix, positions_of_ones, reuse_index);

    return AZ;
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
    for (
        int idx = static_cast<int>(index_min) + 1;
        idx < static_cast<int>(index_max);
        idx++)
    {
        std::vector<size_t> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[hole_to_iterate] = static_cast<size_t>(idx) - 1;

        size_t reuse_index_new = std::min(idx - 1 - hole_to_iterate, reuse_index);

        auto positions_of_ones = calculate_reduce_indices(selected_index_holes_new, num_of_modes);

        Matrix<TScalar> &&L_new = reduce_with_reuse(
            positions_of_ones,
            L,
            reuse_index_new,
            matrix);
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

    auto positions_of_ones = calculate_reduce_indices(selected_index_holes, num_of_modes);
    Matrix<TScalar> &&L_new = reduce_with_reuse(positions_of_ones, L, reuse_index_new, matrix);
    TScalar determinant;
    calc_determinant_cholesky_decomposition(L_new, 2 * reuse_index_new, determinant);

    TScalar partial_torontonian = calculate_partial_torontonian(
        selected_index_holes,
        determinant,
        num_of_modes);

    sum += partial_torontonian;
}

template <typename TScalar>
TScalar torontonian_cpp(Matrix<TScalar> &matrix_in)
{
    size_t dim = matrix_in.rows;
    size_t num_of_modes = dim / 2;

    if (num_of_modes == 0)
        return 1.0;

    Matrix<TScalar> matrix = calc_id_minus_matrix(matrix_in);

    if (num_of_modes == 1)
    {
        TScalar determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        return (TScalar)1.0 / std::sqrt(determinant) - (TScalar)1.0;
    }

    TScalar sum = 0.0;

    // construct the initial selection of the modes
    std::vector<size_t> selected_index_holes;

    // calculate the Cholesky decomposition of the initial matrix to be later reused
    Matrix<TScalar> L = matrix.copy();
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
    iterate_over_selected_modes(selected_index_holes, 0, L, num_of_modes - 1, sum, num_of_modes, matrix);

    // last correction coming from an empty submatrix contribution
    TScalar factor = static_cast<TScalar>((num_of_modes) % 2 ? -1.0 : 1.0);
    sum = sum + factor;

    return sum;
}

template float torontonian_cpp<float>(Matrix<float> &matrix);
template double torontonian_cpp<double>(Matrix<double> &matrix);
