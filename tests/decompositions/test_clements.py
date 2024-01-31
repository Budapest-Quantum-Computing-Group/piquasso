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
from scipy.stats import unitary_group

import piquasso as pq

from piquasso import _math

from piquasso.decompositions.clements import T, Clements


pytestmark = pytest.mark.monkey


@pytest.fixture
def dummy_unitary():
    def func(d):
        return np.array(unitary_group.rvs(d))

    return func


@pytest.fixture(scope="session")
def tolerance():
    return 1e-9


def test_T_beamsplitter_is_unitary():
    theta = np.pi / 3
    phi = np.pi / 4

    beamsplitter = T({"params": [theta, phi], "modes": [0, 1]}, d=2)

    assert _math.linalg.is_unitary(beamsplitter)


def test_eliminate_lower_offdiagonal_2_modes(dummy_unitary, tolerance):
    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 2)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_lower_offdiagonal_3_modes(dummy_unitary, tolerance):
    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_lower_offdiagonal(1, 0)

    beamsplitter = T(operation, 3)

    rotated_U = beamsplitter @ U

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_2_modes(dummy_unitary, tolerance):
    U = dummy_unitary(d=2)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 2)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


def test_eliminate_upper_offdiagonal_3_modes(dummy_unitary, tolerance):
    U = dummy_unitary(d=3)

    decomposition = Clements(U, decompose=False)

    operation = decomposition.eliminate_upper_offdiagonal(1, 0)

    beamsplitter = T.i(operation, 3)

    rotated_U = U @ beamsplitter

    assert np.abs(rotated_U[1, 0]) < tolerance


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_on_n_modes(n, dummy_unitary, tolerance):
    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert np.abs(diagonalized[0, 1]) < tolerance
    assert np.abs(diagonalized[1, 0]) < tolerance

    assert (
        sum(sum(np.abs(diagonalized))) - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_and_composition_on_n_modes(n, dummy_unitary, tolerance):
    U = dummy_unitary(d=n)

    decomposition = Clements(U)

    diagonalized = decomposition.U

    assert (
        sum(sum(np.abs(diagonalized))) - sum(np.abs(np.diag(diagonalized))) < tolerance
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."

    original = Clements.from_decomposition(decomposition)

    assert (U - original < tolerance).all()


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_clements_decomposition_and_composition_on_n_modes_for_identity(n):
    identity = np.identity(n)

    decomposition = Clements(identity)

    diagonalized = decomposition.U

    assert np.isclose(
        sum(sum(np.abs(diagonalized))), sum(np.abs(np.diag(diagonalized)))
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."

    matrix_from_decomposition = Clements.from_decomposition(decomposition)

    assert np.allclose(identity, matrix_from_decomposition)


def test_clements_decomposition_and_composition_on_n_modes_for_matrix_with_0_terms():
    matrix = np.array(
        [
            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [0, 1, 0],
            [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
        ]
    )

    decomposition = Clements(matrix)

    diagonalized = decomposition.U

    assert np.isclose(
        sum(sum(np.abs(diagonalized))), sum(np.abs(np.diag(diagonalized)))
    ), "Absolute sum of matrix elements should be equal to the "
    "diagonal elements, if the matrix is properly diagonalized."

    matrix_from_decomposition = Clements.from_decomposition(decomposition)

    assert np.allclose(matrix, matrix_from_decomposition)


def test_clements_decomposition_using_piquasso_SamplingSimulator(dummy_unitary):
    d = 3
    U = dummy_unitary(d)

    decomposition = Clements(U)

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.StateVector(tuple([1] * d))

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.StateVector(tuple([1] * d))

        for operation in decomposition.inverse_operations:
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=operation["params"][1])
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], 0.0)

        for mode, phaseshift_angle in enumerate(np.angle(decomposition.diagonals)):
            pq.Q(mode) | pq.Phaseshifter(phaseshift_angle)

        for operation in reversed(decomposition.direct_operations):
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], np.pi)
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=-operation["params"][1])

    simulator = pq.SamplingSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_PureFockSimulator(dummy_unitary):
    d = 4
    U = dummy_unitary(d)

    decomposition = Clements(U)

    occupation_numbers = (1, 1, 0, 0)

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.StateVector(occupation_numbers)

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.StateVector(occupation_numbers)

        for operation in decomposition.inverse_operations:
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=operation["params"][1])
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], 0.0)

        for mode, phaseshift_angle in enumerate(np.angle(decomposition.diagonals)):
            pq.Q(mode) | pq.Phaseshifter(phaseshift_angle)

        for operation in reversed(decomposition.direct_operations):
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], np.pi)
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=-operation["params"][1])

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=sum(occupation_numbers) + 1)
    )

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_FockSimulator(dummy_unitary):
    d = 4
    U = dummy_unitary(d)

    decomposition = Clements(U)

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.DensityMatrix((1, 0, 1, 0), (1, 0, 1, 0))

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.DensityMatrix((1, 0, 1, 0), (1, 0, 1, 0))

        for operation in decomposition.inverse_operations:
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=operation["params"][1])
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], 0.0)

        for mode, phaseshift_angle in enumerate(np.angle(decomposition.diagonals)):
            pq.Q(mode) | pq.Phaseshifter(phaseshift_angle)

        for operation in reversed(decomposition.direct_operations):
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], np.pi)
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=-operation["params"][1])

    simulator = pq.FockSimulator(d=d, config=pq.Config(cutoff=3))

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_GaussianSimulator(dummy_unitary):
    d = 3
    U = dummy_unitary(d)

    decomposition = Clements(U)

    squeezings = [0.1, 0.2, 0.3]

    with pq.Program() as program_with_interferometer:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        for operation in decomposition.inverse_operations:
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=operation["params"][1])
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], 0.0)

        for mode, phaseshift_angle in enumerate(np.angle(decomposition.diagonals)):
            pq.Q(mode) | pq.Phaseshifter(phaseshift_angle)

        for operation in reversed(decomposition.direct_operations):
            pq.Q(*operation["modes"]) | pq.Beamsplitter(operation["params"][0], np.pi)
            pq.Q(operation["modes"][0]) | pq.Phaseshifter(phi=-operation["params"][1])

    simulator = pq.GaussianSimulator(d=d, config=pq.Config(cutoff=2))

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition
