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

    hamiltonian = np.block([[-A.conj(), zeros], [zeros, A]])

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
            [0.0, -0.932682, 0.156675, 0.30532, 0.106053, -0.033002],
            [0.932682, 0.0, -0.30532, 0.156675, 0.033002, 0.106053],
            [-0.156675, 0.30532, 0.0, 0.749426, 0.557812, 0.097148],
            [-0.30532, -0.156675, -0.749426, 0.0, -0.097148, 0.557812],
            [-0.106053, -0.033002, -0.557812, 0.097148, 0.0, -0.816744],
            [0.033002, -0.106053, -0.097148, -0.557812, 0.816744, 0.0],
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
            [-0.0, 0.504253, 0.254254, 0.435585, 0.69889, 0.053882],
            [-0.504253, 0.0, -0.444235, 0.643754, -0.259509, 0.258109],
            [-0.254254, 0.444235, 0.0, -0.502168, -0.061072, 0.69434],
            [-0.435585, -0.643754, 0.502168, -0.0, 0.260561, 0.275285],
            [-0.69889, 0.259509, 0.061072, -0.260561, -0.0, -0.610399],
            [-0.053882, -0.258109, -0.69434, -0.275285, 0.610399, 0.0],
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
            [0.0, 0.208302, 0.116836, 0.656998, 0.654506, -0.287986],
            [-0.208302, -0.0, -0.968133, 0.023488, 0.129378, -0.045151],
            [-0.116836, 0.968133, -0.0, -0.147157, -0.165207, -0.010927],
            [-0.656998, -0.023488, 0.147157, -0.0, 0.281808, 0.683177],
            [-0.654506, -0.129378, 0.165207, -0.281808, 0.0, -0.669458],
            [0.287986, 0.045151, 0.010927, -0.683177, 0.669458, 0.0],
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
