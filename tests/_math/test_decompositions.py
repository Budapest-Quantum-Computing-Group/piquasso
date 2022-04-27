#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

import piquasso as pq

from piquasso._math.linalg import is_unitary, is_diagonal
from piquasso._math.symplectic import is_symplectic, xp_symplectic_form
from piquasso._math.decompositions import (
    takagi,
    williamson,
    decompose_to_pure_and_mixed,
)

from piquasso.api.calculator import Calculator
from piquasso._backends.tensorflow.calculator import TensorflowCalculator


@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_real_symmetric_2_by_2_matrix(calculator):
    matrix = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=complex
    )

    singular_values, unitary = takagi(matrix, calculator)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_complex_symmetric_2_by_2_matrix_with_multiplicities(calculator):
    matrix = np.array(
        [
            [1, 2j],
            [2j, 1],
        ],
        dtype=complex,
    )

    singular_values, unitary = takagi(matrix, calculator)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_real_symmetric_3_by_3_matrix(calculator):
    matrix = np.array(
        [
            [1, 2, 3],
            [2, 1, 5],
            [3, 5, 9],
        ],
        dtype=complex
    )

    singular_values, unitary = takagi(matrix, calculator)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_complex_symmetric_3_by_3_matrix(calculator):
    matrix = np.array(
        [
            [1, 2, 3j],
            [2, 1, 5j],
            [3j, 5j, 9],
        ],
    )

    singular_values, unitary = takagi(matrix, calculator)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.monkey
@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_complex_symmetric_6_by_6_matrix_with_multiplicities(
    calculator, generate_unitary_matrix,
):
    singular_values = np.array([1, 1, 2, 2, 2, 3], dtype=complex)

    unitary = generate_unitary_matrix(6)

    matrix = unitary @ np.diag(singular_values) @ unitary.transpose()

    calculated_singular_values, calculated_unitary = takagi(matrix, calculator)

    assert is_unitary(calculated_unitary)
    assert np.allclose(np.abs(calculated_singular_values), calculated_singular_values)
    assert np.allclose(
        matrix,
        calculated_unitary
        @ np.diag(calculated_singular_values)
        @ calculated_unitary.transpose(),
    )


@pytest.mark.monkey
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "calculator", [Calculator(), TensorflowCalculator()]
)
def test_takagi_on_complex_symmetric_N_by_N_matrix(
    N, calculator, generate_complex_symmetric_matrix
):
    matrix = generate_complex_symmetric_matrix(N)
    singular_values, unitary = takagi(matrix, calculator)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


def test_williamson_with_identity():
    covariance_matrix = np.identity(4)
    symplectic, diagonal = williamson(covariance_matrix)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


def test_williamson_with_diagonal_matrix():
    covariance_matrix = np.diag([1, 2, 3, 4])
    symplectic, diagonal = williamson(covariance_matrix)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


def test_williamson_with_squeezed_covariance_matrix():
    d = 3
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=np.pi / 3)
        pq.Q(1, 2) | pq.Squeezing2(r=0.2, phi=np.pi / 5)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    covariance_matrix = state.xpxp_covariance_matrix

    symplectic, diagonal = williamson(covariance_matrix)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


@pytest.mark.monkey
def test_williamson_with_random_positive_definite_matrix(
    generate_random_positive_definite_matrix,
):
    dim = 4
    matrix = generate_random_positive_definite_matrix(dim)

    symplectic, diagonal = williamson(matrix)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(matrix, symplectic @ diagonal @ symplectic.T)


def test_decompose_to_pure_and_mixed_with_identity():
    hbar = 42
    covariance_matrix = hbar * np.identity(4)
    pure_covariance, mixed_contribution = decompose_to_pure_and_mixed(
        covariance_matrix,
        hbar=hbar,
    )

    assert np.allclose(mixed_contribution, 0.0)
    assert np.allclose(covariance_matrix, pure_covariance)


def test_decompose_to_pure_and_mixed_with_pure_gaussian_yield_no_mixed_contribution():
    d = 3
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=np.pi / 3)
        pq.Q(1, 2) | pq.Squeezing2(r=0.2, phi=np.pi / 5)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    covariance_matrix = state.xxpp_covariance_matrix

    pure_covariance, mixed_contribution = decompose_to_pure_and_mixed(
        covariance_matrix,
        hbar=state._config.hbar,
    )

    assert np.allclose(mixed_contribution, 0.0)
    assert np.allclose(covariance_matrix, pure_covariance)


def test_decompose_to_pure_and_mixed_with_reduced_gaussian():
    d = 3
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=np.pi / 3)
        pq.Q(1, 2) | pq.Squeezing2(r=0.2, phi=np.pi / 5)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    reduced_state = state.reduced(modes=(0, 2))

    covariance_matrix = reduced_state.xxpp_covariance_matrix

    pure_covariance, mixed_contribution = decompose_to_pure_and_mixed(
        covariance_matrix,
        hbar=state._config.hbar,
    )

    assert np.allclose(pure_covariance + mixed_contribution, covariance_matrix)
