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

r"""
Multidimensional Hermite recurrence for Gaussian Fock-space amplitudes.

The Fock-space representation of a Gaussian object (pure state, mixed state,
unitary or channel) is characterized by a triple :math:`(A, b, c)`, where
:math:`A` is a complex symmetric matrix, :math:`b` a complex vector and
:math:`c` a complex scalar, see Section III. of
https://arxiv.org/abs/2209.06069.

The (renormalized) amplitudes :math:`\bar{G}^A_\mathbf{k}(b)` satisfy the
order-2 linear recurrence relation (Eq. (45) therein)

.. math::
    \bar{G}_{\mathbf{k} + 1_i} = \frac{1}{\sqrt{k_i + 1}} \left (
        b_i \bar{G}_\mathbf{k}
        + \sum_j \sqrt{k_j} A_{ij} \bar{G}_{\mathbf{k} - 1_j}
    \right ),
    \qquad \bar{G}_\mathbf{0} = c,

which generates *all* amplitudes in a single shared sweep instead of evaluating
each Fock-space matrix element with an independent (loop) hafnian.

For a mixed state on :math:`d` modes the multi-index :math:`\mathbf{k}` is
:math:`2d`-dimensional and splits as ``ket`` :math:`\oplus` ``bra``, so the
density matrix element :math:`\langle \mathrm{bra} | \rho | \mathrm{ket} \rangle`
is exactly :math:`\bar{G}^A_{\mathrm{ket} \oplus \mathrm{bra}}(b)`.
"""

import numpy as np
import numba as nb

from piquasso._math.indices import get_index_in_fock_space


@nb.njit(cache=True)
def _first_nonzero_mode(occupation_numbers: np.ndarray, d: int) -> int:
    """Returns the index of the first nonzero mode (the recurrence pivot)."""
    for mode in range(d):
        if occupation_numbers[mode] > 0:
            return mode

    return 0


@nb.njit(cache=True)
def _entry_raising_ket(
    row: int,
    ket: np.ndarray,
    bra: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    density_matrix: np.ndarray,
    d: int,
) -> complex:
    """Applies Eq. (45) with the pivot in the (nonzero) ``ket`` block."""
    pivot = _first_nonzero_mode(ket, d)

    ket_lowered = ket.copy()
    ket_lowered[pivot] -= 1
    col_lowered = get_index_in_fock_space(ket_lowered)

    value = b[pivot] * density_matrix[row, col_lowered]

    for mode in range(d):
        if ket_lowered[mode] > 0:
            lowered = ket_lowered.copy()
            lowered[mode] -= 1
            value += (
                np.sqrt(ket_lowered[mode])
                * A[pivot, mode]
                * density_matrix[row, get_index_in_fock_space(lowered)]
            )

    for mode in range(d):
        if bra[mode] > 0:
            lowered = bra.copy()
            lowered[mode] -= 1
            value += (
                np.sqrt(bra[mode])
                * A[pivot, d + mode]
                * density_matrix[get_index_in_fock_space(lowered), col_lowered]
            )

    return value / np.sqrt(ket[pivot])


@nb.njit(cache=True)
def _entry_raising_bra(
    bra: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    density_matrix: np.ndarray,
    d: int,
) -> complex:
    """Applies Eq. (45) with the pivot in the ``bra`` block (``ket`` is zero).

    The ``ket``-block sum vanishes identically here, so only the ``bra`` block
    contributes and the column index stays ``0``.
    """
    pivot = _first_nonzero_mode(bra, d)

    bra_lowered = bra.copy()
    bra_lowered[pivot] -= 1
    row_lowered = get_index_in_fock_space(bra_lowered)

    value = b[d + pivot] * density_matrix[row_lowered, 0]

    for mode in range(d):
        if bra_lowered[mode] > 0:
            lowered = bra_lowered.copy()
            lowered[mode] -= 1
            value += (
                np.sqrt(bra_lowered[mode])
                * A[d + pivot, d + mode]
                * density_matrix[get_index_in_fock_space(lowered), 0]
            )

    return value / np.sqrt(bra[pivot])


@nb.njit(cache=True)
def density_matrix_from_gaussian(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    basis: np.ndarray,
) -> np.ndarray:
    r"""Builds a Gaussian density matrix from its :math:`(A, b, c)` triple.

    The element at position ``(i, j)`` equals
    :math:`\langle \mathrm{basis}[i] | \rho | \mathrm{basis}[j] \rangle`, i.e. it
    matches the per-element evaluation with ``bra = basis[i]``, ``ket = basis[j]``
    (``reduce_on = ket`` :math:`\oplus` ``bra``).

    Args:
        A (numpy.ndarray): The :math:`2d \times 2d` complex symmetric matrix.
        b (numpy.ndarray): The :math:`2d` complex vector (zeros for a
            non-displaced state).
        c (complex): The vacuum amplitude :math:`\langle 0 | \rho | 0 \rangle`.
        basis (numpy.ndarray):
            The occupation-number basis of the truncated Fock space, ordered by
            increasing total particle number (as produced by
            :func:`~piquasso._math.fock.get_fock_space_basis`).

    Returns:
        numpy.ndarray: The density matrix in the truncated Fock space.
    """

    d = A.shape[0] // 2
    cardinality = basis.shape[0]

    density_matrix = np.zeros(shape=(cardinality, cardinality), dtype=np.complex128)
    density_matrix[0, 0] = c

    for row in range(cardinality):
        for col in range(cardinality):
            if col > 0:
                density_matrix[row, col] = _entry_raising_ket(
                    row, basis[col], basis[row], A, b, density_matrix, d
                )
            elif row > 0:
                density_matrix[row, col] = _entry_raising_bra(
                    basis[row], A, b, density_matrix, d
                )

    return density_matrix
