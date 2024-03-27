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
def labudde(A):
    """Call to determine the coefficients of the characteristic polynomial using the
    Algorithm 2 of LaBudde method.
    See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
    """
    dim = A.shape[0]

    mtx = A.flatten()

    coeffs = np.zeros(dim * dim, dtype=A.dtype)

    coeffs[0] = -mtx[0]

    coeffs[dim] = coeffs[0] - mtx[dim + 1]

    coeffs[dim + 1] = mtx[0] * mtx[dim + 1] - mtx[1] * mtx[dim]

    beta_prods = np.zeros(dim, dtype=A.dtype)

    for idx in range(2, dim):
        beta_prods[0] = mtx[idx * dim + idx - 1]

        for prod_idx in range(1, idx):
            beta_prods[prod_idx] = (
                beta_prods[prod_idx - 1]
                * mtx[(idx - prod_idx) * dim + (idx - prod_idx - 1)]
            )

        coeffs[idx * dim] = coeffs[(idx - 1) * dim] - mtx[idx * dim + idx]

        for jdx in range(1, idx):
            sum = 0.0

            for mdx in range(1, jdx):
                sum += (
                    mtx[(idx - mdx) * dim + idx]
                    * beta_prods[mdx - 1]
                    * coeffs[(idx - mdx - 1) * dim + jdx - mdx - 1]
                )

            sum += mtx[(idx - jdx) * dim + idx] * beta_prods[jdx - 1]

            coeffs[idx * dim + jdx] = (
                coeffs[(idx - 1) * dim + jdx]
                - mtx[idx * dim + idx] * coeffs[(idx - 1) * dim + jdx - 1]
                - sum
            )

        sum = 0.0

        for mdx in range(1, idx):
            sum += (
                mtx[(idx - mdx) * dim + idx]
                * beta_prods[mdx - 1]
                * coeffs[(idx - mdx - 1) * dim + idx - mdx - 1]
            )

        coeffs[idx * dim + idx] = (
            -mtx[idx * dim + idx] * coeffs[(idx - 1) * dim + idx - 1]
            - sum
            - mtx[idx] * beta_prods[idx - 1]
        )

    for idx in range(dim, dim):
        coeffs[idx * dim] = coeffs[(idx - 1) * dim] - mtx[idx * dim + idx]

        if dim >= 2:
            beta_prods[0] = mtx[idx * dim + idx - 1]

            for prod_idx in range(1, idx):
                beta_prods[prod_idx] = (
                    beta_prods[prod_idx - 1]
                    * mtx[(idx - prod_idx) * dim + (idx - prod_idx - 1)]
                )

            for jdx in range(1, dim):
                sum = 0.0

                for mdx in range(1, jdx):
                    sum += (
                        mtx[(idx - mdx) * dim + idx]
                        * beta_prods[mdx - 1]
                        * coeffs[(idx - mdx - 1) * dim + jdx - mdx - 1]
                    )

                sum += mtx[(idx - jdx) * dim + idx] * beta_prods[jdx - 1]

                coeffs[idx * dim + jdx] = (
                    coeffs[(idx - 1) * dim + jdx]
                    - mtx[idx * dim + idx] * coeffs[(idx - 1) * dim + jdx - 1]
                    - sum
                )

    return coeffs.reshape((dim, dim))
