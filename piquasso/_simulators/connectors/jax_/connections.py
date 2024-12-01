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

"""
This file contains some calculations for `JaxConnector`.
"""

import numpy as np
import numba as nb

from functools import partial

import jax
import jax.numpy as jnp

from piquasso.fermionic._utils import (
    get_fock_subspace_dimension,
    get_fock_subspace_index_first_quantized,
    next_first_quantized,
)

from ..connections import _calculate_index_list_for_appling_interferometer


@nb.njit(cache=True)
def _precalculate_passive_linear_indices(n, d):
    dim = get_fock_subspace_dimension(d, n)

    matrix_row_indices = np.empty(dim, dtype=np.int64)
    deleted_row_indices = np.empty(dim, dtype=np.int64)
    laplace_indices = np.empty((dim, n), dtype=np.int64)
    deleted_indices = np.empty((dim, n), dtype=np.int64)

    first_quantized = np.arange(n)
    for row_idx in range(dim):
        matrix_row_idx = first_quantized[0]
        deleted_row = first_quantized[1:]
        deleted_row_idx = get_fock_subspace_index_first_quantized(deleted_row, d)

        matrix_row_indices[row_idx] = matrix_row_idx
        deleted_row_indices[row_idx] = deleted_row_idx

        for laplace_index in range(n):
            deleted = np.delete(first_quantized, laplace_index)
            deleted_idx = get_fock_subspace_index_first_quantized(deleted, d)
            deleted_indices[row_idx, laplace_index] = deleted_idx
            laplace_indices[row_idx, laplace_index] = first_quantized[laplace_index]

        first_quantized = next_first_quantized(first_quantized, d)

    return (
        matrix_row_indices,
        deleted_row_indices,
        laplace_indices,
        deleted_indices,
    )


@partial(jax.jit, static_argnames="cutoff")
def calculate_interferometer_on_fermionic_fock_space(matrix, cutoff):
    """Calculates the representation of the interferometer matrix on the Fock space.

    This algorithm calculates the determinants of the submatrices recursively using
    Laplace's expansion. For this, it is easiest to work in the first quantized picture.

    Moreover, it is implemented so that the JAX compilation time is minimal, keeping in
    mind that the cutoff is constant during a simulation.
    """
    d = len(matrix)

    subspace_representations = []

    subspace_representations.append(jnp.array([[1.0]], dtype=matrix.dtype))

    if cutoff == 1:
        return subspace_representations

    subspace_representations.append(matrix)

    if cutoff == 2:
        return subspace_representations

    for n in range(2, cutoff):
        (
            matrix_row_indices,
            deleted_row_indices,
            laplace_indices,
            deleted_indices,
        ) = _precalculate_passive_linear_indices(n, d)

        previous_representation = subspace_representations[n - 1]

        signs = jnp.where(jnp.arange(n) % 2 == 0, 1, -1)
        matrix_term = matrix[matrix_row_indices][:, laplace_indices]
        previous_representation_term = previous_representation[deleted_row_indices][
            :, deleted_indices
        ]

        product = signs * matrix_term * previous_representation_term

        representation = jnp.sum(product, axis=2)

        subspace_representations.append(representation)

    return subspace_representations


@partial(jax.jit, static_argnames=["modes", "d", "cutoff"])
def apply_fermionic_passive_linear_to_state_vector(
    representations, state_vector, modes, d, cutoff
):
    """Applies a passive linear gate to a state vector expressed in the Fock basis.

    This function assumes that the n-particle representations of the passive linear gate
    has already been calculated.
    """
    index_list = _calculate_index_list_for_appling_interferometer(modes, d, cutoff)

    new_state_vector = jnp.zeros_like(state_vector)

    for n, indices in enumerate(index_list):
        new_state_vector = new_state_vector.at[indices].set(
            representations[n] @ state_vector[indices]
        )

    return new_state_vector
