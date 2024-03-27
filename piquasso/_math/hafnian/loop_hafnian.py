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

from .powtrace import calculate_power_traces_and_loop_corrections


@nb.njit(cache=True)
def _scale_matrix(matrix):
    if matrix.shape[0] <= 10:
        return matrix, 1.0

    scale_factor = np.sum(np.abs(matrix))

    scale_factor = scale_factor / matrix.shape[0] ** 2 / np.sqrt(2.0)

    inverse_scale_factor = 1 / scale_factor

    for row_idx in range(0, matrix.shape[0]):

        for col_idx in range(0, matrix.shape[1]):
            if col_idx == row_idx:
                matrix[row_idx, col_idx] = matrix[row_idx, col_idx] * np.sqrt(
                    inverse_scale_factor
                )
            else:
                matrix[row_idx, col_idx] = (
                    matrix[row_idx, col_idx] * inverse_scale_factor
                )

    return matrix, scale_factor


@nb.njit(cache=True, parallel=True)
def loop_hafnian(matrix):
    """
    Calculates the loop hafnian of the input matrix using the power trace algorithm
    with Glynn-type iterations.
    """
    if len(matrix) % 2 == 1:
        odd_dim = len(matrix)
        new_matrix = np.empty((odd_dim + 1, odd_dim + 1), dtype=matrix.dtype)
        new_matrix[1:, 1:] = matrix
        new_matrix[0, 0] = 1.0
        new_matrix[1:, 0] = 0.0
        new_matrix[0, 1:] = 0.0

        matrix = new_matrix

    matrix, scale_factor = _scale_matrix(matrix)

    dim = matrix.shape[0]
    dim_over_2 = dim // 2
    permutation_idx_max = 2**dim_over_2

    start_idx = 0
    step_idx = 1
    max_idx = permutation_idx_max // 2

    if matrix.shape[0] == 0:
        return 1.0
    elif matrix.shape[0] % 2 != 0:
        return 0.0

    res = 0.0

    for permutation_idx in nb.prange(start_idx, max_idx, step_idx):
        summand = 0.0

        diag_elements = np.empty(dim, dtype=matrix.dtype)
        cx_diag_elements = np.empty(dim, dtype=matrix.dtype)
        AZ = np.empty((dim, dim), dtype=matrix.dtype)

        fact = False

        for idx in range(0, dim):
            row_offset = idx ^ 1

            for jdx in range(0, dim_over_2):
                neg = (permutation_idx & (1 << jdx)) != 0
                if idx == jdx:
                    fact ^= neg
                if neg:
                    AZ[idx, jdx * 2] = -matrix[row_offset, jdx * 2]
                    AZ[idx, jdx * 2 + 1] = -matrix[row_offset, jdx * 2 + 1]
                else:
                    AZ[idx, jdx * 2] = matrix[row_offset, jdx * 2]
                    AZ[idx, jdx * 2 + 1] = matrix[row_offset, jdx * 2 + 1]

            diag_elements[row_offset] = AZ[idx, row_offset]

            cx_diag_elements[idx] = (
                (-diag_elements[row_offset])
                if (permutation_idx & (1 << (idx // 2))) != 0
                else diag_elements[row_offset]
            )

        traces, loop_corrections = calculate_power_traces_and_loop_corrections(
            cx_diag_elements, diag_elements, AZ, dim_over_2
        )

        aux0 = np.zeros(dim_over_2 + 1, dtype=AZ.dtype)
        aux1 = np.zeros(dim_over_2 + 1, dtype=AZ.dtype)

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

                for kdx in range(idx * jdx + 1, dim_over_2 + 2):
                    data[p_aux1][kdx - 1] += (
                        data[p_aux0][kdx - idx * jdx - 1] * powfactor
                    )

        if fact:
            summand -= data[p_aux1][dim_over_2]
        else:
            summand += data[p_aux1][dim_over_2]

        res += summand

    res *= scale_factor**dim_over_2

    res /= 1 << (dim_over_2 - 1)

    return res
