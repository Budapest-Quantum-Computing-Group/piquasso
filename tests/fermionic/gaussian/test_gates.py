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

    assert np.allclose(
        state.covariance_matrix,
        np.array(
            [
                [0.0, 0.15667544, 0.10605312, -0.93268214, 0.30532025, -0.03300173],
                [-0.15667544, 0.0, 0.55781222, 0.30532025, 0.7494265, 0.09714838],
                [-0.10605312, -0.55781222, -0.0, -0.03300173, 0.09714838, -0.81674436],
                [0.93268214, -0.30532025, 0.03300173, -0.0, 0.15667544, 0.10605312],
                [-0.30532025, -0.7494265, -0.09714838, -0.15667544, 0.0, 0.55781222],
                [0.03300173, -0.09714838, 0.81674436, -0.10605312, -0.55781222, -0.0],
            ]
        ),
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

    assert np.allclose(
        state.covariance_matrix,
        np.array(
            [
                [0.0, 0.25425439, 0.69888954, 0.50425253, 0.43558497, 0.05388169],
                [-0.25425439, 0.0, -0.06107157, 0.44423517, -0.50216753, 0.69433987],
                [-0.69888954, 0.06107157, 0.0, 0.2595091, -0.26056115, -0.61039871],
                [-0.50425253, -0.44423517, -0.2595091, 0.0, 0.64375407, 0.25810896],
                [-0.43558497, 0.50216753, 0.26056115, -0.64375407, 0.0, 0.27528547],
                [-0.05388169, -0.69433987, 0.61039871, -0.25810896, -0.27528547, 0.0],
            ]
        ),
    )


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

    assert np.allclose(
        state.covariance_matrix,
        np.array(
            [
                [0.0, 0.1168362, 0.65450558, 0.20830172, 0.65699774, -0.28798629],
                [-0.1168362, -0.0, -0.16520682, 0.96813296, -0.14715698, -0.010927],
                [-0.65450558, 0.16520682, 0.0, -0.12937805, -0.28180816, -0.66945846],
                [-0.20830172, -0.96813296, 0.12937805, -0.0, 0.0234879, -0.04515084],
                [-0.65699774, 0.14715698, 0.28180816, -0.0234879, 0.0, 0.68317733],
                [0.28798629, 0.010927, 0.66945846, 0.04515084, -0.68317733, -0.0],
            ]
        ),
    )


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
