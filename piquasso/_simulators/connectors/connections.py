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
This file contains some common calculations for `BuiltinConnector`.
"""

import numpy as np
import numba as nb

from functools import lru_cache

from piquasso._math.indices import get_auxiliary_modes

from piquasso.fermionic._utils import (
    get_fock_subspace_dimension,
    cutoff_fock_space_dim_array,
    get_fock_subspace_index_first_quantized,
    next_first_quantized,
    get_fock_space_basis,
    get_fock_space_index,
)


def calculate_interferometer_on_fermionic_fock_space(connector, matrix, cutoff):
    """Calculates the representation of the interferometer matrix on the Fock space.

    This algorithm calculates the determinants of the submatrices recursively using
    Laplace's expansion. For this, it is easiest to work in the first quantized picture.
    """

    np = connector.forward_pass_np
    fallback_np = connector.fallback_np

    d = len(matrix)

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=matrix.dtype))

    if cutoff == 1:
        return subspace_representations

    subspace_representations.append(matrix)

    if cutoff == 2:
        return subspace_representations

    for n in range(2, cutoff):
        dim = get_fock_subspace_dimension(d, n)

        representation = np.zeros(shape=(dim, dim), dtype=matrix.dtype)

        previous_representation = subspace_representations[n - 1]

        first_quantized_row = fallback_np.arange(n)
        for row_idx in range(dim):
            first_quantized_col = fallback_np.arange(n)

            matrix_row_idx = first_quantized_row[0]

            deleted_row = first_quantized_row[1:]
            deleted_row_idx = get_fock_subspace_index_first_quantized(deleted_row, d)
            for col_idx in range(dim):
                sum_ = 0.0
                for laplace_index in range(n):
                    deleted_col = fallback_np.delete(first_quantized_col, laplace_index)
                    deleted_col_idx = get_fock_subspace_index_first_quantized(
                        deleted_col, d
                    )
                    sum_ += (
                        (-1) ** (laplace_index % 2)
                        * matrix[matrix_row_idx, first_quantized_col[laplace_index]]
                        * previous_representation[deleted_row_idx, deleted_col_idx]
                    )

                representation = connector.assign(
                    representation, (row_idx, col_idx), sum_
                )

                first_quantized_col = next_first_quantized(first_quantized_col, d)
            first_quantized_row = next_first_quantized(first_quantized_row, d)

        subspace_representations.append(representation)

    return subspace_representations


@nb.njit(cache=True)
def _nb_calculate_index_list_for_appling_interferometer(modes, d, cutoff):
    subspace = get_fock_space_basis(d=len(modes), cutoff=cutoff)
    auxiliary_subspace = get_fock_space_basis(d=d - len(modes), cutoff=cutoff)

    indices = cutoff_fock_space_dim_array(cutoff=np.arange(cutoff + 1), d=len(modes))
    auxiliary_indices = cutoff_fock_space_dim_array(
        cutoff=np.arange(cutoff + 1), d=d - len(modes)
    )
    auxiliary_modes = get_auxiliary_modes(d, modes)

    all_occupation_numbers = np.zeros(d, dtype=np.int64)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff - n]
        state_index_matrix = np.empty(shape=(size, auxiliary_size), dtype=np.int64)
        for idx1, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[:auxiliary_size]
        ):
            for idx, mode in enumerate(auxiliary_modes):
                all_occupation_numbers[mode] = auxiliary_occupation_numbers[idx]

            for idx2, column_vector_on_subspace in enumerate(n_particle_subspace):
                for idx, mode in enumerate(modes):
                    all_occupation_numbers[mode] = column_vector_on_subspace[idx]

                column_index = get_fock_space_index(all_occupation_numbers)
                state_index_matrix[idx2, idx1] = column_index

        index_list.append(state_index_matrix)

    return index_list


_calculate_index_list_for_appling_interferometer = lru_cache(maxsize=None)(
    _nb_calculate_index_list_for_appling_interferometer
)


def apply_fermionic_passive_linear_to_state_vector(
    connector, representations, state_vector, modes, d, cutoff
):
    """Applies a passive linear gate to a state vector expressed in the Fock basis.

    This function assumes that the n-particle representations of the passive linear gate
    has already been calculated.
    """
    index_list = _calculate_index_list_for_appling_interferometer(modes, d, cutoff)[
        : len(representations)
    ]

    np = connector.forward_pass_np

    new_state_vector = np.zeros_like(state_vector)

    for n, indices in enumerate(index_list):
        new_state_vector = connector.assign(
            new_state_vector, indices, representations[n] @ state_vector[indices]
        )

    return new_state_vector
