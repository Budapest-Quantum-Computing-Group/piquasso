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

import numpy as np
import numba as nb

from piquasso.api.exceptions import InvalidParameter

from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric
from piquasso._math.combinatorics import comb


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


def get_omega(d: int) -> np.ndarray:
    """
    Basis transformation from real to complex basis.
    """

    identity = np.identity(d)

    return np.block(
        [
            [identity, identity],
            [1j * identity, -1j * identity],
        ]
    ) / np.sqrt(2)


@nb.njit(cache=True)
def get_cutoff_fock_space_dimension(d, cutoff):
    sum_ = 0
    for k in range(cutoff):
        sum_ += get_fock_subspace_dimension(d, k)

    return sum_


@nb.njit(cache=True)
def cutoff_fock_space_dim_array(cutoff, d):
    ret = np.empty(cutoff.shape, dtype=np.int64)

    for i in range(len(cutoff)):
        ret[i] = get_cutoff_fock_space_dimension(d, cutoff[i])

    return ret


@nb.njit(cache=True)
def get_fock_subspace_dimension(d, k):
    return comb(d, k)


@nb.njit(cache=True)
def get_fock_space_index(occupation_numbers):
    return _get_fock_space_index_first_quantized(
        _to_first_quantized(occupation_numbers), len(occupation_numbers)
    )


@nb.njit(cache=True)
def get_fock_subspace_index(occupation_numbers):
    return get_fock_subspace_index_first_quantized(
        _to_first_quantized(occupation_numbers), len(occupation_numbers)
    )


@nb.njit(cache=True)
def get_fock_subspace_index_first_quantized(first_quantized, d):
    n = len(first_quantized)

    if n == 0:
        return 0

    sum_ = comb(d, n) - 1

    for i in range(n):
        sum_ -= comb(d - first_quantized[i] - 1, n - i)

    return sum_


@nb.njit(cache=True)
def _get_fock_space_index_first_quantized(first_quantized, d):
    n = len(first_quantized)

    return get_cutoff_fock_space_dimension(
        d, n
    ) + get_fock_subspace_index_first_quantized(first_quantized, d)


@nb.njit(cache=True)
def _to_first_quantized(occupation_numbers):
    n = sum(occupation_numbers)
    first_quantized = np.zeros(n, dtype=nb.int64)

    j = 0
    for i in range(len(occupation_numbers)):
        if occupation_numbers[i] == 1:
            first_quantized[j] = i
            j += 1

    return first_quantized


@nb.njit(cache=True)
def next_first_quantized(first_quantized, d):
    l = len(first_quantized)

    for i in range(l):
        if first_quantized[l - i - 1] < d - i - 1:
            first_quantized[l - i - 1] += 1
            for k in range(l - i, l):
                first_quantized[k] = first_quantized[l - i - 1] + k - l + i + 1

            return first_quantized

    next_first_quantized = np.empty(l + 1, dtype=first_quantized.dtype)

    for i in range(len(next_first_quantized)):
        next_first_quantized[i] = i

    return next_first_quantized


@nb.njit(cache=True)
def next_second_quantized(second_quantized):
    d = len(second_quantized)

    return _to_second_quantized(
        next_first_quantized(_to_first_quantized(second_quantized), d), d
    )


@nb.njit(cache=True)
def _to_second_quantized(first_quantized, d):
    ret = np.zeros(shape=d, dtype=nb.int64)

    for i in range(len(first_quantized)):
        ret[first_quantized[i]] = 1

    return ret


def binary_to_fock_indices(d):
    size = get_cutoff_fock_space_dimension(d, d + 1)

    power_array = np.empty(d, dtype=int)

    for i in range(d):
        power_array[i] = 2 ** (d - 1 - i)

    indices = np.empty(size, dtype=int)

    indices[0] = 0

    index = np.array([], dtype=int)

    for i in range(1, size):
        index = next_first_quantized(index, d)
        second_quantized = _to_second_quantized(index, d)

        indices[i] = second_quantized @ power_array

    return indices


@nb.njit(cache=True)
def get_fock_space_basis(d, cutoff):
    size = get_cutoff_fock_space_dimension(d, cutoff)

    basis = np.empty((size, d), dtype=np.int64)

    basis[0] = np.zeros(d, dtype=np.int64)

    for i in range(1, size):
        basis[i] = next_second_quantized(basis[i - 1])

    return basis
