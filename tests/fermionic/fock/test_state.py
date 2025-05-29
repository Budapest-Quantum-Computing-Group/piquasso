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

for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_PureFockState_d(connector):
    d = 2

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([1, 1])

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    assert state.d == d


@for_all_connectors
def test_PureFockState_fock_probabilities(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1])
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities, [0, np.sin(theta) ** 2, np.cos(theta) ** 2, 0]
    )


@for_all_connectors
def test_PureFockState_norm(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1]) / np.sqrt(2)
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    state = simulator.execute(program).state

    assert np.allclose(state.norm, 1 / 2)


@for_all_connectors
def test_PureFockState_get_particle_detection_probability(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1])
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    state = simulator.execute(program).state

    assert np.allclose(
        state.get_particle_detection_probability([0, 1]), np.cos(theta) ** 2
    )


@for_all_connectors
def test_PureFockState_validate_unnormalized(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1]) / np.sqrt(2)
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.validate()

    assert "The sum of probabilities is" in error.value.args[0]


@for_all_connectors
def test_PureFockState_validate_with_validation_turned_off(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1]) / np.sqrt(2)
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(
        d=2, connector=connector, config=pq.Config(validate=False)
    )

    state = simulator.execute(program).state

    state.validate()


@for_all_connectors
def test_PureFockState_eq_with_itself(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1]) / np.sqrt(2)
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(
        d=2, connector=connector, config=pq.Config(validate=False)
    )

    state = simulator.execute(program).state

    assert state == state


@for_all_connectors
def test_PureFockState_eq_different_type(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1]) / np.sqrt(2)
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(
        d=2, connector=connector, config=pq.Config(validate=False)
    )

    state = simulator.execute(program).state

    some_other_object = object()

    assert state != some_other_object


@for_all_connectors
def test_PureFockState_fock_probabilities_map(connector):
    theta = np.pi / 5

    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector([0, 1])
        pq.Q(0, 1) | pq.Interferometer(U)

    simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

    state = simulator.execute(program).state

    expected = {
        (0, 0): 0,
        (1, 0): np.sin(theta) ** 2,
        (0, 1): np.cos(theta) ** 2,
        (1, 1): 0,
    }

    for occupation_number, probability in expected.items():
        assert np.isclose(state.fock_probabilities_map[occupation_number], probability)
