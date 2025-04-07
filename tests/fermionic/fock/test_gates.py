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

import piquasso as pq

from piquasso.decompositions.clements import clements, instructions_from_decomposition

import numpy as np

import pytest

for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_Interferometer_2_by_2(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1])
        pq.Q(0, 1) | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    fock_state = fock_simulator.execute(program).state

    assert np.allclose(fock_state.state_vector, [0, -np.sin(theta), np.cos(theta), 0])


@pytest.mark.monkey
@for_all_connectors
def test_Interferometer_3_by_3_random(connector, generate_unitary_matrix):
    d = 3
    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        # NOTE: This violates the parity superselection rule, but the simulator should
        # permit it.
        pq.Q(0, 1, 2) | pq.StateVector([0, 0, 1]) / np.sqrt(2)
        pq.Q(0, 1, 2) | pq.StateVector([0, 1, 1]) / np.sqrt(2)
        pq.Q(0, 1, 2) | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    fock_state = fock_simulator.execute(program).state

    assert np.allclose(
        fock_state.state_vector,
        1
        / np.sqrt(2)
        * np.array(
            [
                0.0,
                U[0, 2],
                U[1, 2],
                U[2, 2],
                U[0, 1] * U[1, 2] - U[0, 2] * U[1, 1],
                U[0, 1] * U[2, 2] - U[0, 2] * U[2, 1],
                U[1, 1] * U[2, 2] - U[1, 2] * U[2, 1],
                0.0,
            ]
        ),
    )


@pytest.mark.monkey
@for_all_connectors
def test_Interferometer_clements_equivalence(connector, generate_unitary_matrix):
    d = 3

    U = generate_unitary_matrix(d)

    with pq.Program() as preparation:
        pq.Q() | pq.StateVector(occupation_numbers=[1, 0, 1]) / np.sqrt(2)
        pq.Q() | pq.StateVector(occupation_numbers=[0, 1, 1]) / np.sqrt(2)

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.Interferometer(matrix=U)

    decomposition = clements(U, connector=connector)

    with pq.Program() as decomposed_program:
        pq.Q() | preparation
        decomposed_program.instructions.extend(
            instructions_from_decomposition(decomposition)
        )

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    decomposed_state = simulator.execute(decomposed_program).state

    assert state == decomposed_state


@pytest.mark.monkey
@for_all_connectors
def test_Interferometer_subsystem_equivalence(connector, generate_unitary_matrix):
    d = 3

    U = generate_unitary_matrix(2)

    bigU = np.identity(d, dtype=complex)

    bigU[0, 0] = U[0, 0]
    bigU[0, 1] = U[0, 1]
    bigU[1, 0] = U[1, 0]
    bigU[1, 1] = U[1, 1]

    with pq.Program() as preparation:
        pq.Q() | pq.StateVector(occupation_numbers=[0, 0, 1]) / np.sqrt(3)
        pq.Q() | pq.StateVector(occupation_numbers=[1, 0, 1]) / np.sqrt(3)
        pq.Q() | pq.StateVector(occupation_numbers=[0, 1, 1]) / np.sqrt(3)

    with pq.Program() as program_subsystem:
        pq.Q() | preparation
        pq.Q(0, 1) | pq.Interferometer(matrix=U)

    with pq.Program() as program_full:
        pq.Q() | preparation
        pq.Q(0, 1, 2) | pq.Interferometer(matrix=bigU)

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    state_subsystem = simulator.execute(program_subsystem).state
    state_full = simulator.execute(program_full).state

    assert state_subsystem == state_full


@for_all_connectors
def test_Interferometer_nonconsecutive_ordering_raises_InvalidParameter(connector):
    d = 3

    self_adjoint = np.array(
        [
            [1, 2j, 3 + 4j],
            [-2j, 2, 5],
            [3 - 4j, 5, 6],
        ],
        dtype=complex,
    )
    unitary = connector.expm(1j * self_adjoint)

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    with pq.Program() as preparation:
        pq.Q(0, 1, 2) | pq.StateVector([1, 1, 0])
        pq.Q(0, 2, 1) | pq.Interferometer(unitary)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.execute(preparation)

    assert error.value.args[0] == "Specified modes must be consecutive: modes=(0, 2, 1)"


@for_all_connectors
def test_Squeezing2_on_two_modes(connector):
    d = 2

    r = 0.1
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1])

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )

    state = simulator.execute(program).state

    term_00 = np.sin(r / 2) * np.exp(-1j * phi)
    term_11 = np.cos(r / 2)

    expected_state_vector = np.array([term_00, 0.0, 0.0, term_11])

    assert np.allclose(state.state_vector, expected_state_vector)


@for_all_connectors
def test_Squeezing2_nonconsecutive_ordering_raises_InvalidParameter(connector):
    d = 3

    r = 0.1
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])

        pq.Q(0, 2) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.execute(program)

    assert error.value.args[0] == "Specified modes must be consecutive: modes=(0, 2)"


@for_all_connectors
def test_Squeezing2_cutoff(connector):
    d = 3

    r = 0.1
    phi = np.pi / 7

    cutoff = 2

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0, 0])

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff), connector=connector
    )

    state = simulator.execute(program).state

    expected_state_vector = np.array([np.cos(r / 2), 0.0, 0.0, 0.0])

    assert np.allclose(state.state_vector, expected_state_vector)


@for_all_connectors
def test_ControlledPhase(connector):
    d = 2
    r = 0.1

    phi = np.pi / 7
    cp_phi = np.pi / 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1])

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

        pq.Q(0, 1) | pq.fermionic.ControlledPhase(phi=cp_phi)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )

    state = simulator.execute(program).state

    term_00 = np.sin(r / 2) * np.exp(-1j * phi)
    term_11 = np.cos(r / 2) * np.exp(1j * cp_phi)

    expected_state_vector = np.array([term_00, 0.0, 0.0, term_11])

    assert np.allclose(state.state_vector, expected_state_vector)


@for_all_connectors
def test_IsingXX(connector):
    d = 2

    phi = np.pi / 7
    a = np.sqrt(3 / 4)
    b = np.sqrt(1 / 4)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0]) * a
        pq.Q() | pq.StateVector([0, 1]) * b

        pq.Q(0, 1) | pq.fermionic.IsingXX(phi=phi)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )

    state = simulator.execute(program).state

    expected_state_vector = np.array(
        [
            a * np.cos(phi),
            b * 1j * np.sin(phi),
            b * np.cos(phi),
            a * 1j * np.sin(phi),
        ]
    )

    assert np.allclose(state.state_vector, expected_state_vector)
