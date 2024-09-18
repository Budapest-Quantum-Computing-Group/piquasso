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

from .labudde import labudde
from .hessenberg import (
    transform_matrix_to_hessenberg,
    transform_matrix_to_hessenberg_loop,
)


@nb.njit(cache=True)
def _powtrace_from_charpoly(coeffs_labudde, pow):
    dim = coeffs_labudde.shape[0]

    coeffs = coeffs_labudde.flatten()

    if pow == 0:
        return np.array([dim], dtype=coeffs.dtype)

    traces = np.empty(pow, dtype=coeffs.dtype)

    element_offset = (dim - 1) * dim
    traces[0] = -coeffs[element_offset]

    kdx_max = pow if pow < dim else dim
    for idx in range(2, kdx_max + 1):
        element_offset2 = element_offset + idx - 1
        traces[idx - 1] = -coeffs[element_offset2] * idx

        for j in range(idx - 1, 0, -1):
            traces[idx - 1] -= coeffs[element_offset2 - j] * traces[j - 1]

    element_offset_coeffs = (dim - 1) * dim - 1
    if pow > dim:
        for idx in range(1, pow - dim + 1):
            element_offset = dim + idx - 1
            traces[element_offset] = 0.0

            for jdx in range(1, dim + 1):
                traces[element_offset] -= (
                    traces[element_offset - jdx] * coeffs[element_offset_coeffs + jdx]
                )

    return traces


@nb.njit(cache=True)
def calc_power_traces(A, pow_max):
    """
    Calculate the traces of $A^pow_max$, where `pow_max` is an integer and `A` is a
    square matrix.

    The trace is calculated from the coefficients of its characteristic polynomial.
    In the case that the power `pow_max` is above the size of the matrix we can use an
    optimization described in Appendix B of
    [arxiv:1805.12498](https://arxiv.org/pdf/1104.3769v1.pdf).
    """

    transform_matrix_to_hessenberg(A)

    coeffs_labudde = labudde(A)

    return _powtrace_from_charpoly(coeffs_labudde, pow_max)


@nb.njit(cache=True)
def calculate_power_traces_loop(
    cx_diag_elements,
    diag_elements,
    AZ,
    pow_max,
):
    r"""
    Call to calculate the power traces $Tr(mtx^j)~\forall~1\leq j\leq l$ for a squared
    complex matrix $mtx$ of dimensions $n\times n$ according to Eq. (3.26) of
    [arxiv:1805.12498](https://arxiv.org/pdf/1104.3769v1.pdf).
    """
    transform_matrix_to_hessenberg_loop(AZ, diag_elements, cx_diag_elements)

    coeffs_labudde = labudde(AZ)

    return _powtrace_from_charpoly(coeffs_labudde, pow_max)
