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
    """
    Checks if `H` is a fermionic quadratic Hamiltonian in the Dirac representation.
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
    K = np.diag([1, -1])
    I = np.identity(2)

    ops = [K] * index

    ops.append(op)
    ops += [I] * (d - index - 1)

    assert len(ops) == d

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
    """Calculates the fermionic Hamiltonian.

    It uses Eq. (15) from https://arxiv.org/pdf/2111.08343, but we modified it so that
    the operators will be in lexicographic ordering, otherwise it could be
    anti-lexicographic. One could have implemented this modification differently by
    changing a different convention, but it is easiest to do here.
    """

    d = len(H) // 2

    A = H[d:, d:]
    B = H[:d, d:]

    bigH = connector.np.zeros(shape=(2**d, 2**d), dtype=complex)

    fs, fdags = _get_fs_fdags(d)

    for i in range(d):
        for j in range(d):
            bigH += A[i, j] * fs[i] @ fdags[j]
            bigH += -A[i, j].conj() * fdags[i] @ fs[j]
            bigH += B[i, j] * fdags[i] @ fdags[j]
            bigH += -B[i, j].conj() * fs[i] @ fs[j]

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
