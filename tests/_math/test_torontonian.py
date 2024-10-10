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

import piquasso as pq

import numpy as np

from itertools import chain, combinations


from piquasso._math.torontonian import torontonian, loop_torontonian
from piquasso._math.transformations import from_xxpp_to_xpxp_transformation_matrix


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(iterable, r) for r in range(len(s) + 1))


def torontonian_naive(A: np.ndarray) -> complex:
    """
    Naive torontonian implementation from formulas provided in
    https://arxiv.org/abs/2202.04600.
    """
    d = A.shape[0] // 2

    T = from_xxpp_to_xpxp_transformation_matrix(d)

    A = T.T @ A @ T

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

        summand = factor / np.sqrt(determinant.real + 0.0j)

        ret += summand

    return ret


def loop_torontonian_naive(A, gamma):
    """
    Naive loop torontonian implementation from formulas provided in
    https://arxiv.org/abs/2202.04600.
    """

    d = A.shape[0] // 2

    T = from_xxpp_to_xpxp_transformation_matrix(d)

    A = T.T @ A @ T
    gamma = T.T @ gamma

    if d == 0:
        return 1.0

    ret = 0.0

    for subset in powerset(range(0, d)):
        subset = np.array(subset, dtype=int)
        index = np.ix_(subset, subset)

        vector_index = np.concatenate([subset, subset + d])

        gamma_reduced = gamma[vector_index]

        A_reduced = np.block(
            [
                [A[:d, :d][index], A[:d, d:][index]],
                [A[d:, :d][index], A[d:, d:][index]],
            ]
        )

        inner_mat = np.identity(len(A_reduced)) - A_reduced

        exponential_term = np.exp(
            gamma_reduced @ np.linalg.inv(inner_mat) @ gamma_reduced / 2
        )

        factor = 1.0 if ((d - len(subset)) % 2 == 0) else -1.0

        determinant = np.linalg.det(inner_mat)

        summand = factor * exponential_term / np.sqrt(determinant)

        ret += summand

    return ret


def test_torontonian_empty():
    matrix = np.array([[]], dtype=float)

    assert np.isclose(torontonian(matrix), 1.0)


def test_torontonian_2_by_2_float32():
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
    assert np.isclose(output, 4.0)


def test_torontonian_2_by_2_float64():
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
    assert np.isclose(output, 4.0)


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


@pytest.mark.monkey
@pytest.mark.parametrize("d", (1, 2, 3, 4, 5, 6))
def test_loop_torontonian_random(d, generate_unitary_matrix):
    simulator = pq.GaussianSimulator(d=d)

    program = pq.Program(
        instructions=[pq.Vacuum()]
        + [pq.Displacement(r=np.random.rand()).on_modes(i) for i in range(d)]
        + [pq.Squeezing(r=np.random.rand()).on_modes(i) for i in range(d)]
        + [pq.Interferometer(generate_unitary_matrix(d))]
    )

    state = simulator.execute(program).state

    xpxp_covariance_matrix = state.xpxp_covariance_matrix

    sigma: np.ndarray = (xpxp_covariance_matrix / 2 + np.identity(2 * d)) / 2

    input_matrix = np.identity(len(sigma), dtype=float) - np.linalg.inv(sigma)
    displacement_vector = state.xpxp_mean_vector

    assert np.isclose(
        loop_torontonian(input_matrix, displacement_vector),
        loop_torontonian_naive(input_matrix, displacement_vector),
    )
