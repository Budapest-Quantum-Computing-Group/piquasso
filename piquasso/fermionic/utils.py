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


from piquasso.api.exceptions import InvalidParameter

from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric


import numpy as np


def validate_fermionic_gaussian_hamiltonian(H):
    r"""
    Checks if `H` is a fermionic quadratic Hamiltonian in the Dirac representation.

    More concretely, it validates if :math:`H \in \mathbb{C}^{2d \times 2d}` is a
    self-adjoint matrix of the form

    .. math::
        H = \begin{bmatrix}
            A & -\overline{B} \\
            B & -\overline{A}
        \end{bmatrix}.

    where :math:`A^\dagger = A` and :math:`B^T = - B`.
    """
    d = len(H) // 2

    A = H[d:, d:]
    B = H[:d, d:]

    H11 = H[:d, :d]
    H21 = H[d:, :d]

    if (
        not is_selfadjoint(A)
        or not is_skew_symmetric(B)
        or not np.allclose(-A.conj(), H11)
        or not np.allclose(-B.conj(), H21)
    ):
        raise InvalidParameter("Invalid Hamiltonian specified.")


def get_fermionic_orthogonal_transformation(H, connector):
    np = connector.np
    d = len(H) // 2

    A = H[d:, d:]
    B = H[:d, d:]

    A_plus_B = A + B
    A_minus_B = A - B

    h = np.block(
        [
            [A_plus_B.imag, A_plus_B.real],
            [-A_minus_B.real, A_minus_B.imag],
        ]
    )

    SO = connector.expm(-2 * h)
