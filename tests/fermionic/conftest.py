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


import pytest

import numpy as np

from piquasso.fermionic.gaussian._misc import _get_fs_fdags


@pytest.fixture
def generate_fermionic_gaussian_hamiltonian(
    generate_hermitian_matrix, generate_skew_symmetric_matrix
):
    def func(d):
        A = generate_hermitian_matrix(d)
        B = generate_skew_symmetric_matrix(d)

        return np.block([[-A.conj(), B], [-B.conj(), A]])

    return func


@pytest.fixture
def generate_passive_fermionic_gaussian_hamiltonian(generate_hermitian_matrix):
    def func(d):
        A = generate_hermitian_matrix(d)
        B = np.zeros_like(A)

        return np.block([[-A.conj(), B], [-B.conj(), A]])

    return func


@pytest.fixture
def get_omega():
    """
    :math:`\Omega` in Eq. (10) from https://arxiv.org/pdf/2111.08343.
    """

    def func(d):
        identity = np.identity(d)

        return np.block(
            [
                [identity, identity],
                [1j * identity, -1j * identity],
            ]
        ) / np.sqrt(2)

    return func


@pytest.fixture
def get_majorana_operators():

    def func(d):
        fs, fdags = _get_fs_fdags(d)

        ms = []

        for i in range(d):
            ms.append((fs[i] + fdags[i]) / np.sqrt(2))

        for i in range(d):
            ms.append((fs[i] - fdags[i]) / (1j * np.sqrt(2)))

        return ms

    return func
