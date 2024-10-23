#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numba as nb

from .powtrace import calc_power_traces

from piquasso._math.combinatorics import comb

from .utils import match_occupation_numbers, ix_, get_kept_edges


@nb.njit(cache=True)
def _scale_matrix(matrix):
    r"""
    Scales the matrix for better accuracy.
    """

    dim = matrix.shape[0]
    scale_factor = np.sum(np.abs(matrix) ** 2)

    if scale_factor < 1e-8:
        return matrix, 1.0

    scale_factor = np.sqrt(scale_factor / 2) / (dim**2)

    matrix *= scale_factor

    return matrix, scale_factor


@nb.njit(cache=True)
def _calc_reduced_matrix(matrix: np.ndarray, delta: np.ndarray) -> np.ndarray:
    r"""
    Creates the input data required to calculate power traces according to `delta`.

    Args:
        matrix (np.ndarray): The input matrix.
        delta (np.ndarray): Elements required in $X_{\delta}$ from Eq. (B11) from
            https://arxiv.org/abs/2108.01622.

    Returns:
        Tuple: The input data required to calculate power traces.
    """
    nonzero_indices = np.nonzero(delta)[0]
    nonzero_dim_over_2 = len(nonzero_indices)

    nonzero_dim = 2 * nonzero_dim_over_2

    reduced_matrix = np.empty((nonzero_dim, nonzero_dim), dtype=matrix.dtype)

    for i in range(nonzero_dim_over_2):
        even = 2 * i
        odd = 2 * i + 1

        delta_i = delta[nonzero_indices[i]]

        source_col_even = 2 * nonzero_indices[i] + 1
        source_col_odd = 2 * nonzero_indices[i]

        for row_idx in range(nonzero_dim):
            source_row = 2 * nonzero_indices[row_idx // 2] + row_idx % 2

            reduced_matrix[row_idx, even] = (
                delta_i * matrix[source_row, source_col_even]
            )
            reduced_matrix[row_idx, odd] = delta_i * matrix[source_row, source_col_odd]

    return reduced_matrix


@nb.njit(cache=True)
def _calc_f(traces, scale_factor):
    r"""Calcualates `f` from Appendix B.1 from https://arxiv.org/abs/2108.01622.

    This function uses the previously calculated power traces.

    Args:
        traces (numpy.ndarray): The power traces.
        scale_factor (float): The scale factor of the input matrix for the power traces.
    """

    dim_over_2 = len(traces)
    dim = 2 * dim_over_2

    aux0 = np.zeros(dim_over_2 + 1, dtype=traces.dtype)
    aux1 = np.zeros(dim_over_2 + 1, dtype=traces.dtype)

    aux0[0] = 1.0

    data = [aux0, aux1]

    p_aux0 = 0
    p_aux1 = 1

    inverse_scale_factor = 1 / scale_factor
    for idx in range(1, dim_over_2 + 1):
        factor = traces[idx - 1] * inverse_scale_factor / (2.0 * idx)

        powfactor = 1.0

        if idx % 2 == 1:
            p_aux0 = 0
            p_aux1 = 1
        else:
            p_aux0 = 1
            p_aux1 = 0

        data[p_aux1] = np.copy(data[p_aux0])

        for jdx in range(1, dim // (2 * idx) + 1):
            powfactor = powfactor * factor / jdx
            start_idx = idx * jdx

            for kdx in range(start_idx, dim_over_2 + 1):
                data[p_aux1][kdx] += data[p_aux0][kdx - start_idx] * powfactor

        inverse_scale_factor = inverse_scale_factor / scale_factor

    return data[p_aux1]


@nb.njit(cache=True, parallel=True)
def hafnian_with_reduction(matrix_orig, occupation_numbers):
    r"""
    Calculates the hafnian of the input matrix using the power trace algorithm with
    Glynn-type iterations.

    The algorithm is enhanced by factoring in repetitions according to
    https://arxiv.org/abs/2108.01622.
    """
    n = sum(occupation_numbers)

    if n == 0:
        return 1.0
    elif n % 2 != 0:
        return 0.0

    all_edges, edge_indices = match_occupation_numbers(occupation_numbers)

    matrix_reduced = ix_(matrix_orig, edge_indices, edge_indices)

    if matrix_reduced.shape[0] <= 10:
        scale_factor = 1.0
        matrix = matrix_reduced
    else:
        scale_factor = (
            np.sum(np.abs(matrix_reduced)) / matrix_reduced.shape[0] ** 2 / np.sqrt(2.0)
        )
        matrix = matrix_reduced / scale_factor

    dim_over_2 = np.sum(all_edges)
    number_of_reps = len(all_edges)

    result = 0.0

    size = np.prod(all_edges + 1) // 2

    comb_cache = {}

    for n in range(max(all_edges) + 1):
        for k in range(n + 1):
            comb_cache[(n, k)] = comb(n, k)

    for permutation_idx in nb.prange(size):
        kept_edges = get_kept_edges(all_edges, permutation_idx)
        fact = False
        combinatorial_factor = 1.0

        delta = np.empty_like(all_edges)

        for i in range(number_of_reps):
            no_of_edges = all_edges[i]
            no_of_kept_edges = kept_edges[i]

            fact ^= (no_of_edges - no_of_kept_edges) % 2

            comb_input = (no_of_edges, no_of_kept_edges)
            combinatorial_factor *= comb_cache[comb_input]

            delta[i] = 2 * no_of_kept_edges - no_of_edges

        prefactor = (-1 if fact else 1) * combinatorial_factor

        reduced_matrix = _calc_reduced_matrix(matrix, delta)

        reduced_matrix, second_scale_factor = _scale_matrix(reduced_matrix)

        traces = calc_power_traces(reduced_matrix, dim_over_2)

        summand = prefactor * _calc_f(traces, second_scale_factor)[dim_over_2]

        result += summand

    result *= np.power(scale_factor, dim_over_2)
    result /= 1 << (dim_over_2 - 1)

    return result


@nb.njit(cache=True, parallel=True)
def hafnian_with_reduction_batch(matrix_orig, occupation_numbers, cutoff):
    r"""
    Calculates the hafnian of the input matrix using the power trace algorithm with
    Glynn-type iterations with batching, as described in
    https://arxiv.org/abs/2108.01622.
    """
    particle_number_sum = sum(occupation_numbers)

    odd = particle_number_sum % 2

    if odd:
        result_size = cutoff // 2 - 1
        new_occupation_numbers = np.copy(occupation_numbers)
        new_occupation_numbers[-1] += 1
    else:
        result_size = (cutoff - 1) // 2
        new_occupation_numbers = occupation_numbers

    all_edges, edge_indices = match_occupation_numbers(new_occupation_numbers)

    new_all_edges = np.empty(len(all_edges) + 1, dtype=all_edges.dtype)
    new_all_edges[: len(all_edges)] = all_edges
    new_all_edges[-1] = result_size
    all_edges = new_all_edges

    new_edge_indices = np.empty(len(edge_indices) + 2, dtype=edge_indices.dtype)
    new_edge_indices[: len(edge_indices)] = edge_indices
    new_edge_indices[-1] = len(matrix_orig) - 1
    new_edge_indices[-2] = len(matrix_orig) - 1
    edge_indices = new_edge_indices

    matrix_reduced = ix_(matrix_orig, edge_indices, edge_indices)

    if matrix_reduced.shape[0] <= 10:
        scale_factor = 1.0
        matrix = matrix_reduced
    else:
        scale_factor = (
            np.sum(np.abs(matrix_reduced)) / matrix_reduced.shape[0] ** 2 / np.sqrt(2.0)
        )
        matrix = matrix_reduced / scale_factor

    dim_over_2 = np.sum(all_edges)
    number_of_reps = len(all_edges)

    size = np.prod(all_edges + 1) // 2

    comb_cache = {}

    for n in range(max(all_edges) + 1):
        for k in range(n + 1):
            comb_cache[(n, k)] = comb(n, k)

    result = np.zeros(shape=result_size + 1, dtype=matrix.dtype)

    for permutation_idx in nb.prange(size):
        summand = np.zeros(shape=result_size + 1, dtype=matrix.dtype)
        kept_edges = get_kept_edges(all_edges, permutation_idx)
        fact = False ^ (result_size % 2)
        combinatorial_factor = 1.0

        delta = np.empty_like(all_edges)

        for i in range(number_of_reps):
            no_of_edges = all_edges[i]
            no_of_kept_edges = kept_edges[i]

            fact ^= (no_of_edges - no_of_kept_edges) % 2

            if i != number_of_reps - 1:
                comb_input = (no_of_edges, no_of_kept_edges)
                combinatorial_factor *= comb_cache[comb_input]

            delta[i] = 2 * no_of_kept_edges - no_of_edges

        reduced_matrix = _calc_reduced_matrix(matrix, delta)

        reduced_matrix, second_scale_factor = _scale_matrix(reduced_matrix)

        traces = calc_power_traces(reduced_matrix, dim_over_2)

        fs = _calc_f(traces, second_scale_factor)

        for idx in range(kept_edges[-1], result_size + 1):
            prefactor = (
                (-1 if fact ^ (idx % 2) else 1)
                * combinatorial_factor
                * comb_cache[idx, kept_edges[-1]]
            )

            if idx >= result_size - kept_edges[-1]:
                prefactor *= (
                    1
                    + (-1 if (result_size - idx) % 2 else 1)
                    * comb_cache[idx, result_size - kept_edges[-1]]
                    / comb_cache[idx, kept_edges[-1]]
                )

            summand[idx] = prefactor * fs[dim_over_2 - result_size + idx]

        result += summand

    for i in range(result_size + 1):
        result[i] *= np.power(scale_factor, dim_over_2 + i - result_size)
        result[i] /= 1 << (dim_over_2 - result_size + i)

    concat_result = np.zeros(cutoff, dtype=result.dtype)

    for i in range(len(result)):
        concat_result[2 * i + odd] = result[i]

    if particle_number_sum == 0:
        concat_result[0] = 1.0

    return concat_result
