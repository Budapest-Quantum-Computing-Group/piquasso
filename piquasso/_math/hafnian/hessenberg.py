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


@nb.njit(cache=True)
def _get_reflection_vector(input):
    size = len(input)

    reflect_vector = np.empty(size, dtype=input.dtype)
    norm_v_sqr = 0.0
    for i in range(0, size):
        element = input[i]
        norm_v_sqr += np.real(element) ** 2 + np.imag(element) ** 2
        reflect_vector[i] = element

    sigma = np.sqrt(norm_v_sqr)

    abs_val = np.abs(reflect_vector[0])
    norm_v_sqr = 2 * (norm_v_sqr + abs_val * sigma)

    if abs_val != 0:
        addend = reflect_vector[0] / abs_val * sigma
        reflect_vector[0] += addend
    else:
        reflect_vector[0] += sigma

    if norm_v_sqr == 0.0:
        return reflect_vector, 0.0

    norm_v = np.sqrt(norm_v_sqr)

    reflect_vector /= norm_v

    return reflect_vector, 1.0


HOUSEHOLDER_CUTOFF = 40


@nb.njit(cache=True)
def _mult_a_bconj(a, b):
    return a * np.conj(b)


@nb.njit(cache=True)
def _calc_vH_times_A(A, v, vH_times_A):
    if A.shape[1] > HOUSEHOLDER_CUTOFF:

        cols_mid = A.shape[1] // 2
        A1 = A[:, :cols_mid]
        vH_times_A_1 = vH_times_A[:cols_mid]
        _calc_vH_times_A(A1, v, vH_times_A_1)

        A2 = A[:, cols_mid:]
        vH_times_A_2 = vH_times_A[cols_mid:]

        _calc_vH_times_A(A2, v, vH_times_A_2)

    elif A.shape[0] > HOUSEHOLDER_CUTOFF:

        rows_mid = A.shape[0] // 2
        A1 = A[:rows_mid]
        v1 = v[:rows_mid]

        _calc_vH_times_A(A1, v1, vH_times_A)

        A2 = A[rows_mid:]
        v2 = v[rows_mid:]
        _calc_vH_times_A(A2, v2, vH_times_A)
    else:
        sizeH = len(v)

        for row_idx in range(0, sizeH - 1, 2):
            data_A = A[row_idx]
            data_A2 = A[row_idx + 1]

            for j in range(0, A.shape[1] - 1, 2):
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(data_A[j], v[row_idx])
                vH_times_A[j + 1] = vH_times_A[j + 1] + _mult_a_bconj(
                    data_A[j + 1], v[row_idx]
                )
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(
                    data_A2[j], v[row_idx + 1]
                )
                vH_times_A[j + 1] = vH_times_A[j + 1] + _mult_a_bconj(
                    data_A2[j + 1], v[row_idx + 1]
                )

            if A.shape[1] % 2 == 1:
                j = A.shape[1] - 1
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(data_A[j], v[row_idx])
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(
                    data_A2[j], v[row_idx + 1]
                )

        if sizeH % 2 == 1:
            row_idx = sizeH - 1

            data_A = A[row_idx]

            for j in range(0, A.shape[1] - 1, 2):
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(data_A[j], v[row_idx])
                vH_times_A[j + 1] = vH_times_A[j + 1] + _mult_a_bconj(
                    data_A[j + 1], v[row_idx]
                )

            if A.shape[1] % 2 == 1:
                j = A.shape[1] - 1
                vH_times_A[j] = vH_times_A[j] + _mult_a_bconj(data_A[j], v[row_idx])


@nb.njit(cache=True)
def _calc_vov_times_A(A, v, vH_times_A):
    if A.shape[1] > HOUSEHOLDER_CUTOFF:

        cols_mid = A.shape[1] // 2
        A1 = A[:, :cols_mid]
        vH_times_A_1 = vH_times_A[:cols_mid]

        _calc_vov_times_A(A1, v, vH_times_A_1)

        A2 = A[:, cols_mid:]
        vH_times_A_2 = vH_times_A[cols_mid:]

        _calc_vov_times_A(A2, v, vH_times_A_2)

    elif A.shape[0] > HOUSEHOLDER_CUTOFF:
        rows_mid = A.shape[0] // 2
        A1 = A[:rows_mid]
        v1 = v[:rows_mid]

        _calc_vov_times_A(A1, v1, vH_times_A)

        A2 = A[rows_mid:]
        v2 = v[rows_mid:]
        _calc_vov_times_A(A2, v2, vH_times_A)

    else:
        size_v = len(v)

        for row_idx in range(0, size_v - 1, 2):
            data_A = A[row_idx]
            data_A2 = A[row_idx + 1]

            factor = v[row_idx] * 2.0
            factor2 = v[row_idx + 1] * 2.0

            for kdx in range(0, A.shape[1] - 1, 2):
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx]
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx]
                data_A[kdx + 1] = data_A[kdx + 1] - factor * vH_times_A[kdx + 1]
                data_A2[kdx + 1] = data_A2[kdx + 1] - factor2 * vH_times_A[kdx + 1]

            if A.shape[1] % 2 == 1:
                kdx = A.shape[1] - 1
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx]
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx]

        if size_v % 2 == 1:

            row_idx = len(v) - 1

            data_A = A[row_idx]

            factor = v[row_idx] * 2.0

            for kdx in range(0, A.shape[1] - 1, 2):
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx]
                data_A[kdx + 1] = data_A[kdx + 1] - factor * vH_times_A[kdx + 1]

            if A.shape[1] % 2 == 1:
                kdx = A.shape[1] - 1
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx]


@nb.njit(cache=True)
def _apply_householder_rows(A, v):
    """
    Apply householder transformation on a matrix `A' = (1 - 2*v o v/v^2) A` for one
    specific reflection vector `v`.
    """
    vH_times_A = np.zeros(shape=(A.shape[1]), dtype=A.dtype)

    _calc_vH_times_A(A, v, vH_times_A)

    _calc_vov_times_A(A, v, vH_times_A)


@nb.njit(cache=True)
def _apply_householder_cols_req(A, v):
    """
    Apply householder transformation on a matrix `A' = A(1 - 2*v o v)` for one
    specific reflection vector `v`.
    """
    sizeH = len(v)

    for idx in range(0, A.shape[0] - 1, 2):
        data_A = A[idx]
        data_A2 = A[idx + 1]

        factor = 0.0
        factor2 = 0.0
        for v_idx in range(0, sizeH):
            factor = factor + data_A[v_idx] * v[v_idx]
            factor2 = factor2 + data_A2[v_idx] * v[v_idx]

        factor = factor * 2.0
        factor2 = factor2 * 2.0
        for jdx in range(0, sizeH):
            data_A[jdx] = data_A[jdx] - _mult_a_bconj(factor, v[jdx])
            data_A2[jdx] = data_A2[jdx] - _mult_a_bconj(factor2, v[jdx])

    if A.shape[0] % 2 == 1:
        data_A = A[-1]

        factor = 0.0
        for v_idx in range(0, sizeH):
            factor = factor + data_A[v_idx] * v[v_idx]

        factor = factor * 2.0
        for jdx in range(0, sizeH):
            data_A[jdx] = data_A[jdx] - _mult_a_bconj(factor, v[jdx])


@nb.njit(cache=True)
def transform_matrix_to_hessenberg(mtx):
    """Reduce a general matrix to upper Hessenberg form."""
    for idx in range(1, len(mtx) - 1):
        reflect_vector, norm_v_sqr = _get_reflection_vector(mtx[idx:, idx - 1])

        if norm_v_sqr == 0.0:
            continue

        mtx_strided = mtx[idx:, (idx - 1) :]
        _apply_householder_rows(mtx_strided, reflect_vector)

        mtx_strided = mtx[:, idx:]
        _apply_householder_cols_req(mtx_strided, reflect_vector)


@nb.njit(cache=True)
def transform_matrix_to_hessenberg_loop(mtx, Lv, Rv):
    for idx in range(1, len(mtx) - 1):
        reflect_vector, norm_v_sqr = _get_reflection_vector(mtx[idx:, idx - 1])

        if norm_v_sqr == 0.0:
            continue

        mtx_strided = mtx[idx:, (idx - 1) :]
        _apply_householder_rows(mtx_strided, reflect_vector)

        Lv_strided = Lv[None, idx:]
        _apply_householder_cols_req(Lv_strided, reflect_vector)

        mtx_strided = mtx[:, idx:]
        _apply_householder_cols_req(mtx_strided, reflect_vector)

        Rv_strided = Rv[idx:, None]
        _apply_householder_rows(Rv_strided, reflect_vector)
