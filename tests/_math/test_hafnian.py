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

import jax.numpy as jnp

from scipy.linalg import block_diag
from piquasso._math.hafnian import (
    hafnian_with_reduction,
    hafnian_with_reduction_batch,
    loop_hafnian_with_reduction,
    loop_hafnian_with_reduction_batch,
)
from piquasso._math.jax.hafnian import (
    loop_hafnian_with_reduction as jax_loop_hafnian_with_reduction,
)

from piquasso._math.linalg import reduce_


def test_hafnian_with_empty_reduction():
    matrix = np.array(
        [
            [1, -500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian_with_reduction(matrix, np.array([0, 0])), 1.0)


def test_loop_hafnian_with_empty_reduction():
    matrix = np.array(
        [
            [1, -500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(matrix, np.array([0, 0]), np.array([0, 0])), 1.0
    )


def test_hafnian_with_reduction_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian_with_reduction(matrix, np.array([2, 4])), 18000108)


def test_hafnian_with_reduction_on_2_by_2_complex_matrix():
    matrix = np.array(
        [
            [1, -500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian_with_reduction(matrix, np.array([4, 0])), 3)


def test_loop_hafnian_with_reduction_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.4, 0.6]),
            occupation_numbers=np.array([2, 4]),
        ),
        19097766.06393626,
    )


def test_loop_hafnian_with_reduction_on_1_by_1_complex_matrix():
    matrix = np.array([[1 + 1j]], dtype=complex)

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([1j]),
            occupation_numbers=np.array([6]),
        ),
        -16 - 45j,
    )


def test_loop_hafnian_with_reduction_on_2_by_2_complex_matrix():
    matrix = np.array(
        [
            [1, 500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.4, 0.6]),
            occupation_numbers=np.array([4, 0]),
        ),
        3.9856,
    )


def test_loop_hafnian_with_reduction_on_2_by_2_complex_matrix_8_particles():
    A = np.array([[1 + 2j, 3 + 4j], [3 + 4j, 5 + 6j]])

    diagonal = np.array([-1 - 2j, 3])

    occupation_numbers = np.array([4, 4])

    assert np.isclose(
        loop_hafnian_with_reduction(A, diagonal, occupation_numbers), 12252 - 5184j
    )


def test_loop_hafnian_with_reduction_on_3_by_3_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3],
            [2, 6, 7],
            [3, 7, 3],
        ],
        dtype=float,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.1, 0.2, 0.3]),
            occupation_numbers=np.array([1, 3, 1]),
        ),
        51.28824,
    )


def test_loop_hafnian_with_reduction_on_3_by_3_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j],
            [2, 6, 7],
            [3j, 7, 3],
        ],
        dtype=complex,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.1, 0.2j, 0.3]),
            occupation_numbers=np.array([1, 3, 1]),
        ),
        12.468 + 16.90776j,
    )


def test_hafnian_with_reduction_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [3, 7, 3, 4],
            [4, 8, 4, 8],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian_with_reduction(matrix, np.array([2, 1, 0, 3])), 1344)


def test_hafnian_with_reduction_on_4_by_4_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert np.isclose(hafnian_with_reduction(matrix, np.array([2, 1, 0, 3])), 960j)


def test_loop_hafnian_with_reduction_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [3, 7, 3, 4],
            [4, 8, 4, 8],
        ],
        dtype=float,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.1, 0.2, 0.3, 0.4]),
            occupation_numbers=np.array([2, 1, 0, 4]),
        ),
        3226.0762112,
    )


def test_loop_hafnian_with_reduction_on_4_by_4_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.array([0.1, 0.2j, 0.3, 0.4 + 0.1j]),
            occupation_numbers=np.array([1, 3, 1, 1]),
        ),
        205.376424 + 690.048304j,
    )


def test_hafnian_with_reduction_on_6_by_6_real_matrix():
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

    assert np.isclose(hafnian_with_reduction(matrix, np.ones(6, dtype=int)), 1262.0)


def test_hafnian_with_reduction_on_6_by_6_complex_matrix():
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

    assert np.isclose(
        hafnian_with_reduction(matrix, np.array([1, 1, 1, 1, 0, 2])), -22 + 1192j
    )


def test_loop_hafnian_with_reduction_on_6_by_6_real_matrix():
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

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.zeros(6),
            occupation_numbers=np.ones(6, dtype=int),
        ),
        1262,
    )


def test_loop_hafnian_with_reduction_on_6_by_6_complex_matrix():
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

    assert np.isclose(
        loop_hafnian_with_reduction(
            matrix,
            np.zeros(6),
            occupation_numbers=np.array([1, 1, 1, 1, 0, 2]),
        ),
        -22.0 + 1192.0j,
    )


@pytest.mark.monkey
def test_random_10_by_10_hafnian_with_reduction(generate_symmetric_matrix):
    matrix = generate_symmetric_matrix(10)

    hafnian_with_reduction(matrix, np.random.randint(0, 2, 10))


@pytest.mark.monkey
def test_hafnian_with_reduction_of_complex_symmetric_matrix_with_odd_dimension_is_zero(
    generate_complex_symmetric_matrix,
):
    matrix = generate_complex_symmetric_matrix(3)

    assert np.isclose(hafnian_with_reduction(matrix, np.ones(3, dtype=int)), 0.0)


@pytest.mark.monkey
def test_loop_hafnian_with_reduction_of_complex_symmetric_block_diagonal_matrix(
    generate_complex_symmetric_matrix,
):
    occupation_numbers = np.random.randint(0, 2, 5)

    submatrix = generate_complex_symmetric_matrix(5)
    submatrix_loop_hafnian = loop_hafnian_with_reduction(
        submatrix, np.zeros(5), occupation_numbers
    )

    matrix = block_diag(submatrix, np.conj(submatrix))
    matrix_loop_hafnian = loop_hafnian_with_reduction(
        matrix, np.zeros(10), np.concatenate([occupation_numbers, occupation_numbers])
    )

    assert np.isclose(
        np.conj(submatrix_loop_hafnian) * submatrix_loop_hafnian,
        matrix_loop_hafnian,
    )


def test_hafnian_with_reduction_6_particles():
    A = np.array(
        [
            [0.06918447, 0.50455312, 0.52623749, 0.51119496],
            [0.56996393, 0.62138659, 0.42424946, 0.6582932],
            [0.79820529, 0.36861047, 0.91219376, 0.18621944],
            [0.37366373, 0.3069308, 0.29075932, 0.60748435],
        ]
    )

    A = A + A.T

    haf = hafnian_with_reduction(A, np.array([2, 0, 1, 3]))

    assert np.isclose(haf, 11.024591504142506)


def test_hafnian_with_reduction_12_particles():
    A = np.array(
        [
            [0.06918447, 0.50455312, 0.52623749, 0.51119496],
            [0.56996393, 0.62138659, 0.42424946, 0.6582932],
            [0.79820529, 0.36861047, 0.91219376, 0.18621944],
            [0.37366373, 0.3069308, 0.29075932, 0.60748435],
        ]
    )

    A = A + A.T

    haf = hafnian_with_reduction(A, np.array([6, 3, 2, 3]))

    assert np.isclose(haf, 37655.53201446679)


def test_hafnian_with_reduction_12_particles_with_0():
    A = np.array(
        [
            [0.06918447, 0.50455312, 0.52623749, 0.51119496],
            [0.56996393, 0.62138659, 0.42424946, 0.6582932],
            [0.79820529, 0.36861047, 0.91219376, 0.18621944],
            [0.37366373, 0.3069308, 0.29075932, 0.60748435],
        ]
    )

    A = A + A.T

    haf = hafnian_with_reduction(A, np.array([7, 4, 0, 3]))

    assert np.isclose(haf, 13629.353597466445)


@pytest.mark.monkey
def test_hafnian_with_reduction_equivalence():
    for _ in range(10):
        d = 5
        max_photons = 4
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        occupation_numbers = np.random.randint(0, max_photons, d)

        expected = hafnian_with_reduction(
            reduce_(A, occupation_numbers),
            np.ones(sum(occupation_numbers), dtype=int),
        )
        actual = hafnian_with_reduction(A, occupation_numbers)

        assert np.isclose(expected, actual)


@pytest.mark.monkey
def test_hafnian_with_reduction_scaling_equivalence():
    for _ in range(10):
        d = np.random.randint(1, 10)
        max_photons = np.random.randint(1, 10)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        occupation_numbers = np.random.randint(0, max_photons, d)

        scaling_factor = np.random.rand() + 1 / 2

        scaled_before = hafnian_with_reduction(scaling_factor * A, occupation_numbers)
        scaled_after = scaling_factor ** (
            np.sum(occupation_numbers) // 2
        ) * hafnian_with_reduction(A, occupation_numbers)

        assert np.isclose(scaled_before, scaled_after)


@pytest.mark.monkey
def test_loop_hafnian_with_reduction_equivalence():
    for _ in range(100):
        d = 5
        max_photons = 6
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        occupation_numbers = np.random.randint(0, max_photons, d)

        diagonal = np.random.rand(d) + 1j * np.random.rand(d)

        reduced_A = reduce_(A, occupation_numbers)

        reduced_diagonal = reduce_(diagonal, occupation_numbers)

        np.fill_diagonal(reduced_A, reduced_diagonal)

        expected = loop_hafnian_with_reduction(
            reduced_A,
            np.diag(reduced_A),
            np.ones(sum(occupation_numbers), dtype=int),
        )
        actual = loop_hafnian_with_reduction(A, diagonal, occupation_numbers)

        assert np.isclose(expected, actual)


@pytest.mark.monkey
def test_loop_hafnian_with_reduction_scaling_equivalence():
    for _ in range(10):
        d = np.random.randint(1, 10)
        max_photons = np.random.randint(1, 5)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        diagonal = np.random.rand(d) + 1j * np.random.rand(d)

        occupation_numbers = 2 * np.random.randint(0, max_photons, d)

        scaling_factor = np.random.rand() + 1 / 2

        scaled_before = loop_hafnian_with_reduction(
            scaling_factor * A, np.sqrt(scaling_factor) * diagonal, occupation_numbers
        )
        scaled_after = scaling_factor ** (
            np.sum(occupation_numbers) // 2
        ) * loop_hafnian_with_reduction(A, diagonal, occupation_numbers)

        assert np.isclose(scaled_before, scaled_after)


@pytest.mark.monkey
def test_loop_hafnian_with_reduction_odd_even_equivalence():
    for _ in range(100):
        d = np.random.randint(1, 10)
        max_photons = np.random.randint(1, 5)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A_odd = A + A.T

        diagonal_odd = np.random.rand(d) + 1j * np.random.rand(d)

        occupation_numbers_odd = 2 * np.random.randint(0, max_photons, d)
        occupation_numbers_odd[0] += 1

        odd_case = loop_hafnian_with_reduction(
            A_odd, diagonal_odd, occupation_numbers_odd
        )

        occupation_numbers_even = np.concatenate([[1], occupation_numbers_odd])
        diagonal_even = np.concatenate([[1.0], diagonal_odd])

        A_even = block_diag([1.0], A_odd)

        even_case = loop_hafnian_with_reduction(
            A_even, diagonal_even, occupation_numbers_even
        )

        assert np.isclose(odd_case, even_case)


@pytest.mark.parametrize(
    "occupation_numbers, cutoff",
    [
        (np.array([1, 3, 0, 0]), 5),
        (np.array([1, 3, 2, 0]), 6),
        (np.array([1, 5, 2, 0]), 9),
        (np.array([1, 5, 2, 0]), 11),
        (np.array([2, 0, 2, 0]), 12),
        (np.array([0, 0, 0, 0]), 7),
        (np.array([2, 2, 4, 0]), 6),
        (np.array([0, 1, 0, 0]), 8),
        (np.array([2, 1, 0, 0]), 7),
        (np.array([2, 1, 4, 0]), 12),
    ],
)
def test_hafnian_with_reduction_batch(occupation_numbers, cutoff):
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )
    mask = np.array([0, 0, 0, 1], dtype=int)

    assert np.allclose(
        hafnian_with_reduction_batch(matrix, occupation_numbers, cutoff),
        [
            hafnian_with_reduction(matrix, occupation_numbers + i * mask)
            for i in range(cutoff)
        ],
    )


@pytest.mark.monkey
def test_hafnian_with_reduction_batch_random():
    for _ in range(100):
        d = np.random.randint(1, 10)
        max_photons = np.random.randint(1, 5)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)

        A = A + A.T

        occupation_numbers = np.random.randint(0, max_photons, d)
        occupation_numbers[-1] = 0

        cutoff = np.random.randint(1, 12)

        mask = np.zeros_like(occupation_numbers)
        mask[-1] = 1

        assert np.allclose(
            hafnian_with_reduction_batch(A, occupation_numbers, cutoff),
            [
                hafnian_with_reduction(A, occupation_numbers + i * mask)
                for i in range(cutoff)
            ],
        )


@pytest.mark.parametrize(
    "occupation_numbers, cutoff",
    [
        (np.array([1, 3, 0, 0]), 5),
        (np.array([1, 3, 2, 0]), 6),
        (np.array([1, 5, 2, 0]), 9),
        (np.array([1, 5, 2, 0]), 11),
        (np.array([2, 0, 2, 0]), 12),
        (np.array([0, 0, 0, 0]), 7),
        (np.array([2, 2, 4, 0]), 6),
        (np.array([0, 1, 0, 0]), 8),
        (np.array([2, 1, 0, 0]), 7),
        (np.array([2, 1, 4, 0]), 12),
        (np.array([2, 4, 5, 0]), 16),
        (np.array([1, 1, 1, 0]), 15),
    ],
)
def test_loop_hafnian_with_reduction_batch(occupation_numbers, cutoff):
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )
    mask = np.array([0, 0, 0, 1], dtype=int)
    diagonal = np.array([0.1, 0.2, 0.3, 0.4], dtype=complex)

    assert np.allclose(
        loop_hafnian_with_reduction_batch(matrix, diagonal, occupation_numbers, cutoff),
        [
            loop_hafnian_with_reduction(matrix, diagonal, occupation_numbers + i * mask)
            for i in range(cutoff)
        ],
    )


@pytest.mark.monkey
def test_loop_hafnian_with_reduction_batch_random():
    for _ in range(100):
        d = np.random.randint(1, 12)
        max_photons = np.random.randint(1, 5)
        A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
        A = A + A.T
        diagonal = np.random.rand(d) + 1j * np.random.rand(d)

        occupation_numbers = np.random.randint(0, max_photons, d)
        occupation_numbers[-1] = 0

        cutoff = np.random.randint(1, 12)

        mask = np.zeros_like(occupation_numbers)
        mask[-1] = 1

        expected = [
            loop_hafnian_with_reduction(A, diagonal, occupation_numbers + i * mask)
            for i in range(cutoff)
        ]
        actual = loop_hafnian_with_reduction_batch(
            A, diagonal, occupation_numbers, cutoff
        )

        assert np.allclose(expected, actual)


def test_jax_loop_hafnian_with_reduction():
    matrix = jnp.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert jnp.isclose(
        jax_loop_hafnian_with_reduction(
            matrix,
            jnp.array([0.1, 0.2j, 0.3, 0.4 + 0.1j]),
            jnp.array([1, 3, 1, 1]),
        ),
        205.376424 + 690.048304j,
    )
