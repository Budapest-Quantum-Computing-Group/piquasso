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

import pytest
import numpy as np
from scipy.linalg import schur
from piquasso._math.pfaffian import pfaffian


def pfaffian_naive(matrix):
    if matrix.shape[0] == 0:
        return 1.0

    if matrix.shape[0] % 2 == 1:
        return 0.0

    blocks, O = schur(matrix)
    a = np.diag(blocks, 1)[::2]

    return np.prod(a) * np.linalg.det(O)


def test_pfaffian_empty():
    matrix = np.array([[]], dtype=float)

    assert np.isclose(pfaffian(matrix), 1.0)


def test_pfaffian_2_by_2_skew_symmetric_float32():
    matrix = np.array(
        [
            [0, 1.7],
            [-1.7, 0],
        ],
        dtype=np.float32,
    )

    result = pfaffian_naive(matrix)
    output = pfaffian(matrix)

    assert output.dtype == np.float32

    assert np.isclose(output, result)
    assert np.isclose(output, 1.7)


def test_pfaffian_2_by_2_skew_symmetric_float64():
    matrix = np.array(
        [
            [0, 1.7],
            [-1.7, 0],
        ],
        dtype=np.float64,
    )

    output = pfaffian(matrix)

    assert output.dtype == np.float64

    assert np.isclose(output.item(), pfaffian_naive(matrix))
    assert np.isclose(output.item(), 1.7)


@pytest.mark.monkey
def test_pfaffian_4_by_4_skew_symmetric_random():
    for _ in range(100):

        A = np.random.rand(4, 4)
        matrix = A - A.T

        result = pfaffian_naive(matrix)
        output = pfaffian(matrix)

        assert np.isclose(output, result)


@pytest.mark.monkey
def test_pfaffian_6_by_6_skew_symmetric_random():
    for _ in range(100):

        A = np.random.rand(6, 6)
        matrix = A - A.T

        result = pfaffian_naive(matrix)
        output = pfaffian(matrix)

        assert np.isclose(output, result)
