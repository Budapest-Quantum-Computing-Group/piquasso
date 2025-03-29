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

import piquasso as pq

from scipy.linalg import expm

from piquasso.decompositions.clements import clements, instructions_from_decomposition


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@pytest.mark.monkey
@for_all_connectors
def test_GaussianHamiltonian_and_Interferometer_equivalence(
    connector,
    generate_hermitian_matrix,
):
    d = 3

    A = generate_hermitian_matrix(d)
    zeros = np.zeros_like(A)

    unitary = expm(-2j * A.conj())

    hamiltonian = np.block([[A, zeros], [zeros, -A.conj()]])

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    with pq.Program() as passive_program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.Interferometer(matrix=unitary)

    state_1 = simulator.execute(program).state
    state_2 = simulator.execute(passive_program).state

    assert state_1 == state_2


@pytest.mark.monkey
@for_all_connectors
def test_passive_GaussianHamiltonian_preserves_particle_number(
    connector,
    generate_passive_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_passive_fermionic_gaussian_hamiltonian(d)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    mean_particle_numberss = state.mean_particle_numbers(modes=(0, 1, 2))

    assert (
        mean_particle_numberss.dtype == simulator.config.dtype
    ), "Particle number should be real-valued"
    assert np.isclose(
        sum(mean_particle_numberss),
        sum(state_vector),
    )


@pytest.mark.monkey
@for_all_connectors
def test_GaussianHamiltonian_subsystem_equivalence(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    modes = (0, 2)

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d - 1)

    embedded_hamiltonian = connector.np.zeros((2 * d, 2 * d), dtype=hamiltonian.dtype)

    doubled_modes = np.concatenate([modes, np.array(modes) + d])

    embedded_hamiltonian = connector.assign(
        embedded_hamiltonian, np.ix_(doubled_modes, doubled_modes), hamiltonian
    )

    state_vector = [1, 0, 1]
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    with pq.Program() as program_subsystem:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q(*modes) | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    with pq.Program() as program_embedded:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=embedded_hamiltonian)

    subsystem_state = simulator.execute(program_subsystem).state
    embedded_state = simulator.execute(program_embedded).state

    assert subsystem_state == embedded_state


@for_all_connectors
def test_Interferometer_on_state_vector(connector):
    passive_hamiltonian = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])

    U = expm(1j * passive_hamiltonian)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.Interferometer(U)

    simulator = pq.fermionic.GaussianSimulator(d=3, connector=connector)

    state = simulator.execute(program).state

    expected_covariance_matrix = np.array(
        [
            [0.0, -0.932682, -0.156675, 0.30532, -0.106053, -0.033002],
            [0.932682, 0.0, -0.30532, -0.156675, 0.033002, -0.106053],
            [0.156675, 0.30532, 0.0, 0.749426, -0.557812, 0.097148],
            [-0.30532, 0.156675, -0.749426, 0.0, -0.097148, -0.557812],
            [0.106053, -0.033002, 0.557812, 0.097148, 0.0, -0.816744],
            [0.033002, 0.106053, -0.097148, 0.557812, 0.816744, 0.0],
        ]
    )

    assert np.allclose(
        state.covariance_matrix,
        expected_covariance_matrix,
    )


@for_all_connectors
def test_Interferometer(connector):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    passive_hamiltonian = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])

    U = expm(1j * passive_hamiltonian)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers=state_vector)

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

        pq.Q() | pq.Interferometer(matrix=U)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    expected_covariance_matrix = np.array(
        [
            [0.0, 0.469966, -0.19859, 0.780816, 0.228407, 0.279019],
            [-0.469966, 0.0, -0.66337, 0.08203, -0.493825, -0.297456],
            [0.19859, 0.66337, 0.0, -0.392254, -0.478024, 0.371661],
            [-0.780816, -0.08203, 0.392254, 0.0, -0.069497, 0.474242],
            [-0.228407, 0.493825, 0.478024, 0.069497, 0.0, -0.686026],
            [-0.279019, 0.297456, -0.371661, -0.474242, 0.686026, 0.0],
        ]
    )

    assert np.allclose(state.covariance_matrix, expected_covariance_matrix)


@for_all_connectors
def test_Interferometer_on_subsystem(connector):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    passive_hamiltonian = np.array([[1, 2j], [-2j, 5]])

    U = expm(1j * passive_hamiltonian)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers=state_vector)

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

        pq.Q(0, 2) | pq.Interferometer(matrix=U)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    expected_covariance_matrix = np.array(
        [
            [0.0, 0.16447, 0.252142, 0.588858, 0.365978, -0.654737],
            [-0.16447, 0.0, -0.951866, 0.162249, 0.145479, -0.139325],
            [-0.252142, 0.951866, -0.0, -0.147157, -0.040228, 0.084272],
            [-0.588858, -0.162249, 0.147157, -0.0, 0.672212, 0.391659],
            [-0.365978, -0.145479, 0.040228, -0.672212, 0.0, -0.625626],
            [0.654737, 0.139325, -0.084272, -0.391659, 0.625626, 0.0],
        ]
    )

    assert np.allclose(state.covariance_matrix, expected_covariance_matrix)


@pytest.mark.monkey
@for_all_connectors
def test_Beamsplitter_preserves_particle_number(connector):
    d = 3

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter(theta=-np.pi / 5, phi=np.pi / 4)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    assert np.isclose(
        sum(state.mean_particle_numbers(modes=(0, 1, 2))),
        sum(state_vector),
    )


@for_all_connectors
def test_5050_Beamsplitter(connector):
    d = 2
    state_vector = [1, 0]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

        pq.Q() | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    assert np.allclose(
        state.correlation_matrix,
        np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, -0.5],
                [0.0, 0.0, -0.5, 0.5],
            ]
        ),
    )


@for_all_connectors
def test_Interferometer_clements_equivalence(connector):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    passive_hamiltonian = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])

    U = expm(1j * passive_hamiltonian)

    with pq.Program() as preparation:
        pq.Q() | pq.StateVector(occupation_numbers=[1, 0, 1])
        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.Interferometer(matrix=U)

    decomposition = clements(U, connector=connector)

    with pq.Program() as decomposed_program:
        pq.Q() | preparation
        decomposed_program.instructions.extend(
            instructions_from_decomposition(decomposition)
        )

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    decomposed_state = simulator.execute(decomposed_program).state

    assert state == decomposed_state


@for_all_connectors
def test_Interferometer_subsystem_equivalence(connector, generate_unitary_matrix):
    d = 3

    U = generate_unitary_matrix(2)

    bigU = np.identity(d, dtype=complex)

    bigU[0, 0] = U[0, 0]
    bigU[0, 2] = U[0, 1]
    bigU[2, 0] = U[1, 0]
    bigU[2, 2] = U[1, 1]

    with pq.Program() as preparation:
        pq.Q() | pq.StateVector(occupation_numbers=[1, 1, 0])

    with pq.Program() as program_subsystem:
        pq.Q() | preparation
        pq.Q(0, 2) | pq.Interferometer(matrix=U)

    with pq.Program() as program_full:
        pq.Q() | preparation
        pq.Q(0, 1, 2) | pq.Interferometer(matrix=bigU)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state_subsystem = simulator.execute(program_subsystem).state
    state_full = simulator.execute(program_full).state

    assert state_subsystem == state_full


@for_all_connectors
def test_Squeezing2_on_two_modes_00(connector):
    d = 2

    state_vector = [0, 0]

    r = 0.1
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers=state_vector)

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    term_00 = np.cos(r / 2)
    term_11 = -np.sin(r / 2) * np.exp(1j * phi)

    expected_state_vector = np.array([term_00, 0.0, 0.0, term_11])

    expected_density_matrix = np.outer(
        expected_state_vector, expected_state_vector.conj()
    )

    actual_density_matrix = state.density_matrix

    assert np.allclose(expected_density_matrix, actual_density_matrix)


@for_all_connectors
def test_Squeezing2_on_two_modes_11(connector):
    d = 2

    state_vector = [1, 1]

    r = 0.1
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers=state_vector)

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    term_00 = np.sin(r / 2) * np.exp(-1j * phi)
    term_11 = np.cos(r / 2)

    expected_state_vector = np.array([term_00, 0.0, 0.0, term_11])

    expected_density_matrix = np.outer(
        expected_state_vector, expected_state_vector.conj()
    )

    actual_density_matrix = state.density_matrix

    assert np.allclose(expected_density_matrix, actual_density_matrix)


@for_all_connectors
def test_Squeezing2_leaves_odd_occupation_numbers_invariant(connector):
    d = 2

    state_vector = [1, 0]

    r = 0.1
    phi = np.pi / 7

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    with pq.Program() as empty_program:
        pq.Q() | pq.StateVector(occupation_numbers=state_vector)

    with pq.Program() as program:
        pq.Q() | empty_program

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    initial_state = simulator.execute(empty_program).state
    squeezed_state = simulator.execute(program).state

    assert initial_state == squeezed_state
