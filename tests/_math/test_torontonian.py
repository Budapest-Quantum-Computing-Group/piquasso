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

from piquasso._math.torontonian import torontonian
from piquasso._math.combinatorics import powerset


def torontonian_naive(A: np.ndarray) -> complex:
    d = A.shape[0] // 2

    if d == 0:
        return 1.0 + 0j

    ret = 0.0 + 0j

    for subset in powerset(range(0, d)):
        index = np.ix_(subset, subset)

        A_reduced = np.block(
            [
                [A[:d, :d][index], A[:d, d:][index]],
                [A[d:, :d][index], A[d:, d:][index]],
            ]
        )

        factor = 1.0 if ((d - len(subset)) % 2 == 0) else -1.0

        inner_mat = np.identity(len(A_reduced)) - A_reduced

        determinant = np.linalg.det(inner_mat)

        summand = factor / np.sqrt(
           determinant.real + 0.0j
        )

        ret += summand

    return ret


def test_torontonian_empty():
    matrix = np.array([[]], dtype=float)

    assert np.isclose(torontonian(matrix), 1.0)


def test_torontonian_2_by_2_float64():
    matrix = np.array(
        [
            [0.6, 0.4],
            [0.4, 0.5],
        ],
        dtype=np.float32,
    )

    output = torontonian(matrix)

    assert output.dtype == np.float32

    assert np.isclose(output, torontonian_naive(matrix))
    assert np.isclose(output, 4.000001430511475)


def test_torontonian_2_by_2_float128():
    matrix = np.array(
        [
            [0.6, 0.4],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )

    output = torontonian(matrix)

    assert output.dtype == np.float64

    assert np.isclose(output, torontonian_naive(matrix))
    assert np.isclose(output, 4.000001430511475)


@pytest.mark.monkey
def test_torontonian_4_by_4_random():
    A = np.random.rand(4, 4)
    matrix = A @ A.T

    matrix /= max(np.linalg.eigvals(matrix)) + 1.0

    torontonian(matrix)

    assert np.isclose(torontonian(matrix), torontonian_naive(matrix))


@pytest.mark.monkey
def test_torontonian_6_by_6_random():
    A = np.random.rand(6, 6)
    matrix = A @ A.T

    matrix /= max(np.linalg.eigvals(matrix)) + 1.0

    torontonian(matrix)

    assert np.isclose(torontonian(matrix), torontonian_naive(matrix))
