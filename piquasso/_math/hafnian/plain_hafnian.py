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


@nb.njit(cache=True)
def _scale_matrix(AZ):
    dim = AZ.shape[0]

    flattened_AZ = AZ.flatten()
    scale_factor_AZ = 0.0
    size_AZ = dim * dim
    for idx in range(0, size_AZ):
        scale_factor_AZ += np.abs(flattened_AZ[idx]) ** 2

    if scale_factor_AZ < 1e-8:
        scale_factor_AZ = 1.0
    else:
        scale_factor_AZ = np.sqrt(scale_factor_AZ / 2) / size_AZ
        for idx in range(0, size_AZ):
            flattened_AZ[idx] *= scale_factor_AZ

    return flattened_AZ.reshape(dim, dim), scale_factor_AZ


@nb.njit(cache=True, parallel=True)
def hafnian(mtx_orig):
    """
    Calculates the hafnian of the input matrix using the power trace algorithm with
    Glynn-type iterations.
    """
    if mtx_orig.shape[0] == 0:
        return 1.0

    elif mtx_orig.shape[0] % 2 != 0:
        return 0.0

    if mtx_orig.shape[0] <= 10:
        mtx = mtx_orig
        scale_factor = 1.0
    else:
        scale_factor = np.sum(np.abs(mtx_orig))

        scale_factor = scale_factor / mtx_orig.shape[0] ** 2 / np.sqrt(2.0)

        mtx = mtx_orig / scale_factor

    start_idx = 0
    step_idx = 1

    dim = mtx.shape[0]
    dim_over_2 = dim // 2
    max_idx = 2 ** (dim_over_2 - 1)

    res = 0.0

    for permutation_idx in nb.prange(start_idx, max_idx, step_idx):
        summand = 0.0

        B = np.empty((dim, dim), dtype=mtx.dtype)

        fact = False
        for idx in range(0, dim):
            row_offset = idx ^ 1
            for jdx in range(0, dim_over_2):
                neg = (permutation_idx & (1 << jdx)) != 0
                if idx == jdx:
                    fact ^= neg

                if neg:
                    B[idx, 2 * jdx] = -mtx[row_offset, jdx * 2]
                    B[idx, 2 * jdx + 1] = -mtx[row_offset, jdx * 2 + 1]
                else:
                    B[idx, 2 * jdx] = mtx[row_offset, jdx * 2]
                    B[idx, 2 * jdx + 1] = mtx[row_offset, jdx * 2 + 1]

        B, scale_factor_B = _scale_matrix(B)

        traces = calc_power_traces(B, dim_over_2)

        aux0 = np.zeros(dim_over_2 + 1, dtype=B.dtype)
        aux1 = np.zeros(dim_over_2 + 1, dtype=B.dtype)

        aux0[0] = 1.0

        data = [aux0, aux1]

        p_aux0 = 0
        p_aux1 = 1

        inverse_scale_factor = 1 / scale_factor_B
        for idx in range(1, dim_over_2 + 1):
            factor = traces[idx - 1] * inverse_scale_factor / (2.0 * idx)

            inverse_scale_factor = inverse_scale_factor / scale_factor_B

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

    res *= np.power(scale_factor, dim_over_2)
    res /= 1 << (dim_over_2 - 1)

    return res
