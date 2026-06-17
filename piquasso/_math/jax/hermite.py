#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
Translation of `piquasso._math.hermite` to JAX.
"""

import numpy as np

import jax
import jax.numpy as jnp

import numba as nb

from piquasso._math.indices import get_index_in_fock_space


@nb.njit(cache=True)
def _precompute_lowered_indices(basis: np.ndarray) -> np.ndarray:
    # NOTE: This part of the code is not JAX-compatible, but it is only run once, so it
    # is not a performance bottleneck. The differentiability is not violated.
    cardinality, d = basis.shape

    lowered_indices = np.zeros(shape=(cardinality, d), dtype=np.int32)

    for index in range(cardinality):
        occupation_numbers = basis[index]

        for mode in range(d):
            if occupation_numbers[mode] == 0:
                continue

            lowered = occupation_numbers.copy()
            lowered[mode] -= 1

            lowered_indices[index, mode] = get_index_in_fock_space(lowered)

    return lowered_indices


def _first_nonzero_mode(occupation_numbers):
    return jnp.argmax(occupation_numbers > 0)


def _entry_raising_ket(row, col, basis, lowered_indices, A, b, density_matrix):
    d = basis.shape[1]
    real_dtype = jnp.real(A).dtype

    ket = basis[col]
    bra = basis[row]

    pivot = _first_nonzero_mode(ket)

    col_lowered = lowered_indices[col, pivot]

    value = b[pivot] * density_matrix[row, col_lowered]

    def ket_body(mode, value):
        occupation = basis[col_lowered, mode]
        lowered_col = lowered_indices[col_lowered, mode]

        return value + (
            jnp.sqrt(occupation.astype(real_dtype))
            * A[pivot, mode]
            * density_matrix[row, lowered_col]
        )

    value = jax.lax.fori_loop(0, d, ket_body, value)

    def bra_body(mode, value):
        occupation = bra[mode]
        lowered_row = lowered_indices[row, mode]

        return value + (
            jnp.sqrt(occupation.astype(real_dtype))
            * A[pivot, d + mode]
            * density_matrix[lowered_row, col_lowered]
        )

    value = jax.lax.fori_loop(0, d, bra_body, value)

    return value / jnp.sqrt(ket[pivot].astype(real_dtype))


def _entry_raising_bra(row, basis, lowered_indices, A, b, density_matrix):
    d = basis.shape[1]
    real_dtype = jnp.real(A).dtype

    bra = basis[row]

    pivot = _first_nonzero_mode(bra)

    row_lowered = lowered_indices[row, pivot]

    value = b[d + pivot] * density_matrix[row_lowered, 0]

    def bra_body(mode, value):
        occupation = basis[row_lowered, mode]
        lowered_row = lowered_indices[row_lowered, mode]

        return value + (
            jnp.sqrt(occupation.astype(real_dtype))
            * A[d + pivot, d + mode]
            * density_matrix[lowered_row, 0]
        )

    value = jax.lax.fori_loop(0, d, bra_body, value)

    return value / jnp.sqrt(bra[pivot].astype(real_dtype))


@jax.jit
def _density_matrix_from_gaussian_core(A, b, c, basis, lowered_indices):
    cardinality = basis.shape[0]

    dtype = A.dtype

    density_matrix = jnp.zeros(
        shape=(cardinality, cardinality),
        dtype=dtype,
    )
    density_matrix = density_matrix.at[0, 0].set(c)

    def row_body(row, density_matrix):
        def col_body(col, density_matrix):
            value = jax.lax.cond(
                col > 0,
                lambda _: _entry_raising_ket(
                    row,
                    col,
                    basis,
                    lowered_indices,
                    A,
                    b,
                    density_matrix,
                ),
                lambda _: jax.lax.cond(
                    row > 0,
                    lambda __: _entry_raising_bra(
                        row,
                        basis,
                        lowered_indices,
                        A,
                        b,
                        density_matrix,
                    ),
                    lambda __: density_matrix[0, 0],
                    operand=None,
                ),
                operand=None,
            )

            return density_matrix.at[row, col].set(value)

        return jax.lax.fori_loop(0, cardinality, col_body, density_matrix)

    return jax.lax.fori_loop(0, cardinality, row_body, density_matrix)


def density_matrix_from_gaussian(
    A: jnp.ndarray,
    b: jnp.ndarray,
    c: complex,
    basis: np.ndarray,
) -> jnp.ndarray:
    lowered_indices = jnp.asarray(_precompute_lowered_indices(basis))

    return _density_matrix_from_gaussian_core(
        jnp.asarray(A),
        jnp.asarray(b),
        jnp.asarray(c),
        jnp.asarray(basis),
        lowered_indices,
    )
