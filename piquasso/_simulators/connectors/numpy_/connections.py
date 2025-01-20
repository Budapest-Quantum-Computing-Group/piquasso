#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

"""
This file contains some common calculations for `NumpyConnector`.
"""

import numpy as np

import numba as nb

from .._utils import precalculate_fermionic_passive_linear_indices


@nb.njit(cache=True)
def calculate_interferometer_on_fermionic_fock_space(matrix, cutoff):
    """Calculates the representation of the interferometer matrix on the Fock space.

    This algorithm calculates the determinants of the submatrices recursively using
    Laplace's expansion. For this, it is easiest to work in the first quantized picture.
    """

    d = len(matrix)

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=matrix.dtype))

    if cutoff == 1:
        return subspace_representations

    subspace_representations.append(matrix)

    if cutoff == 2:
        return subspace_representations

    for n in range(2, cutoff):
        laplace_indices, deleted_indices = (
            precalculate_fermionic_passive_linear_indices(n, d)
        )

        dim = laplace_indices.shape[0]

        representation = np.zeros(shape=(dim, dim), dtype=matrix.dtype)

        previous_representation = subspace_representations[n - 1]

        for row_idx in range(dim):
            matrix_row_idx = laplace_indices[row_idx, 0]
            deleted_row_idx = deleted_indices[row_idx, 0]

            for col_idx in range(dim):
                for laplace_index in range(n):
                    deleted_col_idx = deleted_indices[col_idx, laplace_index]

                    representation[row_idx, col_idx] += (
                        (-1) ** (laplace_index % 2)
                        * matrix[
                            matrix_row_idx, laplace_indices[col_idx, laplace_index]
                        ]
                        * previous_representation[deleted_row_idx, deleted_col_idx]
                    )

        subspace_representations.append(representation)

    return subspace_representations
