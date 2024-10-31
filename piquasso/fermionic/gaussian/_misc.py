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

from typing import TYPE_CHECKING

import numpy as np

from piquasso.api.exceptions import InvalidParameter

from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric

if TYPE_CHECKING:
    from piquasso.api.connector import BaseConnector


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


def tensor_product(ops):
    if len(ops) == 1:
        return ops[0]

    return np.kron(ops[0], tensor_product(ops[1:]))


def _embed_f(op, index, d):
    r"""Embeds the single-mode Dirac operators into the Fock space.

    This function returns with

    .. math::
        f_i &= Z^{\otimes (i-1)} \otimes f \otimes, I^{d-i} \\\\
        f_i^\dagger &= Z^{\otimes (i-1)} \otimes f^\dagger \otimes I^{d-i},

    where :math:`f` and :math:`f^\dagger` are the single-mode annihilation and creation
    operators, respectively.
    """

    Z = np.diag([1, -1])
    I = np.identity(2)

    ops = [Z] * index

    ops.append(op)
    ops += [I] * (d - index - 1)

    return tensor_product(ops)


def _fdag():
    return np.array(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=complex,
    )


def _f():
    return np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=complex,
    )


def _get_fs_fdags(d):
    fs = []
    fdags = []

    for i in range(d):
        fs.append(_embed_f(_f(), i, d))
        fdags.append(_embed_f(_fdag(), i, d))

    return fs, fdags


def get_fermionic_hamiltonian(H, connector):
    r"""Calculates the fermionic Hamiltonian.

    This function returns

    .. math::
        \hat{H} = \mathbf{f} H \mathbf{f}^\dagger,

    where :math:`H \in \mathbb{C}^{2d \times 2d}` is a self-adjoint matrix of the form

    .. math::
        H = \begin{bmatrix}
            A & -\overline{B} \\
            B & -\overline{A}
        \end{bmatrix}.

    where :math:`A^\dagger = A` and :math:`B^T = - B`.
    """

    d = len(H) // 2

    bigH = connector.np.zeros(shape=(2**d, 2**d), dtype=complex)

    fs, fdags = _get_fs_fdags(d)

    f = fs + fdags

    for i in range(2 * d):
        for j in range(2 * d):
            bigH += H[i, j] * f[i] @ f[j].T

    return bigH


def get_omega(d: int, connector: "BaseConnector") -> np.ndarray:
    """
    Basis transformation from real to complex basis.
    """

    np = connector.np

    identity = np.identity(d)

    return np.block(
        [
            [identity, identity],
            [1j * identity, -1j * identity],
        ]
    ) / np.sqrt(2)
