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

import numba as nb
import numpy as np

from piquasso._math.combinatorics import comb

from .powtrace import calculate_power_traces_loop

from .utils import match_occupation_numbers, ix_, get_kept_edges
from .loop_corrections import calculate_loop_corrections


@nb.njit(cache=True)
def _scale_matrix(matrix):
    r"""
    Scales the matrix for better accuracy.
    """

    dim = matrix.shape[0]

    if dim <= 10:
        return matrix, 1.0

    scale_factor = np.sum(np.abs(matrix))

    scale_factor = scale_factor / dim**2 / np.sqrt(2.0)

    inverse_scale_factor = 1 / scale_factor

    for row_idx in range(dim):
        for col_idx in range(dim):
            if col_idx == row_idx:
                matrix[row_idx, col_idx] = matrix[row_idx, col_idx] * np.sqrt(
                    inverse_scale_factor
                )
            else:
                matrix[row_idx, col_idx] = (
                    matrix[row_idx, col_idx] * inverse_scale_factor
                )

    return matrix, scale_factor


@nb.njit(cache=True)
def _calc_B_and_reduced_diagonals(matrix, diagonal, delta):
    r"""
    Creates the input data required to calculate power traces and loop corrections
    according to `delta`.

    Args:
        matrix (np.ndarray): The input matrix.
        diagonal (np.ndarray): The displacement term.
        delta (np.ndarray): Elements required in $X_{\delta}$ from Eq. (B11) from
            https://arxiv.org/abs/2108.01622.

    Returns:
        Tuple: The input data required to calculate power traces and loop corrections.
    """
    nonzero_indices = np.nonzero(delta)[0]
    nonzero_dim_over_2 = len(nonzero_indices)

    nonzero_dim = 2 * nonzero_dim_over_2

    B = np.empty((nonzero_dim, nonzero_dim), dtype=matrix.dtype)
    reduced_diagonal_left = np.empty(nonzero_dim, dtype=matrix.dtype)
    reduced_diagonal_right = np.empty(nonzero_dim, dtype=matrix.dtype)

    for i in range(nonzero_dim_over_2):
        even = 2 * i
        odd = 2 * i + 1

        delta_i = delta[nonzero_indices[i]]

        source_col_even = 2 * nonzero_indices[i] + 1
        source_col_odd = 2 * nonzero_indices[i]

        for row_idx in range(nonzero_dim):
            source_row = 2 * nonzero_indices[row_idx // 2] + row_idx % 2

            B[row_idx, even] = delta_i * matrix[source_row, source_col_even]
            B[row_idx, odd] = delta_i * matrix[source_row, source_col_odd]

        reduced_diagonal_right[even] = diagonal[source_col_odd]
        reduced_diagonal_right[odd] = diagonal[source_col_even]

        reduced_diagonal_left[even] = delta_i * diagonal[source_col_even]
        reduced_diagonal_left[odd] = delta_i * diagonal[source_col_odd]

    return B, reduced_diagonal_left, reduced_diagonal_right


@nb.njit(cache=True)
def _calc_f_loop(traces, loop_corrections):
    r"""Calcualates `f` from Appendix B.1 from https://arxiv.org/abs/2108.01622.

    This function uses the previously calculated power traces and loop corrections.

    Args:
        traces (numpy.ndarray): The power traces.
        loop_corrections (numpy.ndarray): Loop corrections.
    """

    dim_over_2 = len(traces)
    dim = 2 * dim_over_2

    aux0 = np.zeros(dim_over_2 + 1, dtype=traces.dtype)
    aux1 = np.zeros(dim_over_2 + 1, dtype=traces.dtype)

    aux0[0] = 1.0

    data = [aux0, aux1]

    p_aux0 = 0
    p_aux1 = 1

    for idx in range(1, dim_over_2 + 1):
        factor = traces[idx - 1] / (2.0 * idx) + loop_corrections[idx - 1] * 0.5

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

            for kdx in range(idx * jdx, dim_over_2 + 1):
                data[p_aux1][kdx] += data[p_aux0][kdx - idx * jdx] * powfactor

    return data[p_aux1][dim_over_2]


@nb.njit(cache=True, parallel=True)
def loop_hafnian_with_reduction(matrix_orig, diagonal_orig, occupation_numbers):
    r"""
    Calculates the loop hafnian of the input matrix using the power trace algorithm
    with Glynn-type iterations.

    The algorithm is enhanced by factoring in repetitions according to
    https://arxiv.org/abs/2108.01622.
    """

    n = sum(occupation_numbers)

    diagonal_orig = diagonal_orig.astype(matrix_orig.dtype)

    if n == 0:
        return 1.0
    elif n % 2 == 1:
        # Handling the odd case by extending the matrix with a 1 to be even.
        #
        # TODO: This is not the best handling of the odd case, and we can definitely
        # squeeze out a bit more performance if needed.
        dim = len(matrix_orig)
        new_matrix = np.empty((dim + 1, dim + 1), dtype=matrix_orig.dtype)
        new_matrix[1:, 1:] = matrix_orig
        new_matrix[0, 0] = 1.0
        new_matrix[1:, 0] = 0.0
        new_matrix[0, 1:] = 0.0

        matrix_orig = new_matrix

        d = len(occupation_numbers)

        new_occupation_numbers = np.empty(d + 1, dtype=occupation_numbers.dtype)
        new_diagonal = np.empty(d + 1, dtype=diagonal_orig.dtype)

        new_occupation_numbers[0] = 1
        new_diagonal[0] = 1.0

        for i in range(d):
            new_occupation_numbers[i + 1] = occupation_numbers[i]
            new_diagonal[i + 1] = diagonal_orig[i]

        occupation_numbers = new_occupation_numbers
        diagonal_orig = new_diagonal

    all_edges, edge_indices = match_occupation_numbers(occupation_numbers)

    matrix = ix_(matrix_orig, edge_indices, edge_indices)
    diagonal = diagonal_orig[edge_indices]

    matrix, scale_factor = _scale_matrix(matrix)
    diagonal = diagonal / np.sqrt(scale_factor)

    dim_over_2 = sum(all_edges)
    dim = 2 * dim_over_2

    result = 0.0

    comb_cache = {}

    for n in range(max(all_edges) + 1):
        for k in range(n + 1):
            comb_cache[(n, k)] = comb(n, k)

    size = np.prod(all_edges + 1) // 2

    for permutation_idx in nb.prange(size):
        kept_edges = get_kept_edges(all_edges, permutation_idx)
        fact = False
        combinatorial_factor = 1.0

        delta = np.empty_like(all_edges)

        for i in range(len(all_edges)):
            no_of_edges = all_edges[i]
            no_of_kept_edges = kept_edges[i]

            fact ^= (no_of_edges - no_of_kept_edges) % 2
            comb_input = (no_of_edges, no_of_kept_edges)
            combinatorial_factor *= comb_cache[comb_input]

            delta[i] = 2 * no_of_kept_edges - no_of_edges

        prefactor = (-1.0 if fact else 1) * combinatorial_factor

        B, reduced_diagonal_left, reduced_diagonal_right = (
            _calc_B_and_reduced_diagonals(matrix, diagonal, delta)
        )

        traces = calculate_power_traces_loop(
            reduced_diagonal_right, reduced_diagonal_left, B, dim_over_2
        )

        loop_corrections = calculate_loop_corrections(
            reduced_diagonal_right, reduced_diagonal_left, B, dim_over_2
        )

        summand = prefactor * _calc_f_loop(traces, loop_corrections)

        result += summand

    result *= scale_factor**dim_over_2

    result /= 1 << (dim_over_2 - 1)

    return result
