#
# Copyright 2021 Budapest Quantum Computing Group
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

from scipy.linalg import block_diag
from piquasso._math.hafnian import (
    hafnian,
    loop_hafnian,
    hafnian_with_reduction,
    loop_hafnian_with_reduction,
)


def test_hafnian_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 500.0)


def test_hafnian_on_2_by_2_complex_matrix():
    matrix = np.array(
        [
            [1, 500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian(matrix), 500.0j)


def test_loop_hafnian_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(loop_hafnian(matrix), 506.0)


def test_loop_hafnian_on_2_by_2_complex_matrix():
    matrix = np.array(
        [
            [1, 500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(loop_hafnian(matrix), 6.0 + 500.0j)


def test_loop_hafnian_on_3_by_3_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3],
            [2, 6, 7],
            [3, 7, 3],
        ],
        dtype=float,
    )

    assert np.isclose(loop_hafnian(matrix), 49.0)


def test_loop_hafnian_on_3_by_3_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j],
            [2, 6, 7],
            [3j, 7, 3],
        ],
        dtype=complex,
    )

    assert np.isclose(loop_hafnian(matrix), 6 + 43j)


def test_hafnian_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [3, 7, 3, 4],
            [4, 8, 4, 8],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 60.0)


def test_hafnian_on_4_by_4_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian(matrix), 12.0)


def test_loop_hafnian_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [3, 7, 3, 4],
            [4, 8, 4, 8],
        ],
        dtype=float,
    )

    assert np.isclose(loop_hafnian(matrix), 572.0)


def test_loop_hafnian_on_4_by_4_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert np.isclose(loop_hafnian(matrix), -284.0 + 72.0j)


def test_hafnian_on_6_by_6_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 6, 7, 8, 9, 5],
            [3, 7, 3, 4, 3, 7],
            [4, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2, 0],
            [6, 5, 7, 1, 0, 1],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 1262.0)


def test_hafnian_on_6_by_6_complex_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4j, 5, 6j],
            [2, 6, 7j, 8, 9, 5],
            [3, 7j, 3, 4, 3, 7],
            [4j, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2j, 0],
            [6j, 5, 7, 1, 0, 1],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian(matrix), 387.0 + 707.0j)


def test_loop_hafnian_on_6_by_6_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 6, 7, 8, 9, 5],
            [3, 7, 3, 4, 3, 7],
            [4, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2, 0],
            [6, 5, 7, 1, 0, 1],
        ],
        dtype=float,
    )

    assert np.isclose(loop_hafnian(matrix), 15195.0)


def test_loop_hafnian_on_6_by_6_complex_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4j, 5, 6j],
            [2, 6, 7j, 8, 9, 5],
            [3, 7j, 3, 4, 3, 7],
            [4j, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2j, 0],
            [6j, 5, 7, 1, 0, 1],
        ],
        dtype=complex,
    )

    assert np.isclose(loop_hafnian(matrix), 2238.0 + 5273.0j)


@pytest.mark.monkey
def test_random_10_by_10_hafnian(generate_symmetric_matrix):
    matrix = generate_symmetric_matrix(10)

    assert hafnian(matrix)


@pytest.mark.monkey
def test_hafnian_of_complex_symmetric_matrix_with_odd_dimension_is_zero(
    generate_complex_symmetric_matrix,
):
    matrix = generate_complex_symmetric_matrix(3)

    assert np.isclose(hafnian(matrix), 0.0)


@pytest.mark.monkey
def test_loop_hafnian_of_complex_symmetric_block_diagonal_matrix(
    generate_complex_symmetric_matrix,
):
    submatrix = generate_complex_symmetric_matrix(5)
    submatrix_loop_hafnian = loop_hafnian(submatrix)

    matrix = block_diag(submatrix, submatrix.conj())
    matrix_loop_hafnian = loop_hafnian(matrix)

    assert np.isclose(
        submatrix_loop_hafnian.conj() * submatrix_loop_hafnian,
        matrix_loop_hafnian,
    )


def test_loop_hafnian_and_hafnian_with_zero_reduction():

    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    reduce_on = (1, 2)

    assert np.isclose(
        hafnian_with_reduction(matrix, reduce_on),
        loop_hafnian_with_reduction(matrix, np.zeros(len(matrix)), reduce_on),
    )
