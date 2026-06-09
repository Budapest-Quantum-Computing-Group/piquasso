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
def density_matrix_from_a_b_c(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    basis: np.ndarray,
) -> np.ndarray:
    r"""Builds a Gaussian density matrix from its :math:`(A, b, c)` triple.

    The element at position ``(i, j)`` equals
    :math:`\langle \mathrm{basis}[i] | \rho | \mathrm{basis}[j] \rangle`, matching
    ``DensityMatrixCalculation.get_density_matrix_element`` with ``bra = basis[i]``,
    ``ket = basis[j]`` (i.e. ``reduce_on = ket ⊕ bra``).

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
        bra = basis[row]
        for col in range(cardinality):
            if row == 0 and col == 0:
                continue

            ket = basis[col]

            if col > 0:
                # The combined index has a nonzero `ket` block: raise there.
                pivot = 0
                for index in range(d):
                    if ket[index] > 0:
                        pivot = index
                        break

                ket_lowered = ket.copy()
                ket_lowered[pivot] -= 1
                col_lowered = get_index_in_fock_space(ket_lowered)

                value = b[pivot] * density_matrix[row, col_lowered]

                for index in range(d):
                    if ket_lowered[index] > 0:
                        ket_lowered2 = ket_lowered.copy()
                        ket_lowered2[index] -= 1
                        value += (
                            np.sqrt(ket_lowered[index])
                            * A[pivot, index]
                            * density_matrix[row, get_index_in_fock_space(ket_lowered2)]
                        )

                for index in range(d):
                    if bra[index] > 0:
                        bra_lowered = bra.copy()
                        bra_lowered[index] -= 1
                        value += (
                            np.sqrt(bra[index])
                            * A[pivot, d + index]
                            * density_matrix[
                                get_index_in_fock_space(bra_lowered), col_lowered
                            ]
                        )

                density_matrix[row, col] = value / np.sqrt(ket[pivot])
            else:
                # `ket` block is zero (col == 0): raise in the `bra` block. The
                # `ket`-block sum in the recurrence vanishes identically.
                pivot = 0
                for index in range(d):
                    if bra[index] > 0:
                        pivot = index
                        break

                bra_lowered = bra.copy()
                bra_lowered[pivot] -= 1
                row_lowered = get_index_in_fock_space(bra_lowered)

                value = b[d + pivot] * density_matrix[row_lowered, col]

                for index in range(d):
                    if bra_lowered[index] > 0:
                        bra_lowered2 = bra_lowered.copy()
                        bra_lowered2[index] -= 1
                        value += (
                            np.sqrt(bra_lowered[index])
                            * A[d + pivot, d + index]
                            * density_matrix[get_index_in_fock_space(bra_lowered2), col]
                        )

                density_matrix[row, col] = value / np.sqrt(bra[pivot])

    return density_matrix
