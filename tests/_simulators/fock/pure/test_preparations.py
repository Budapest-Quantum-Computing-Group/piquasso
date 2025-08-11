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


def test_create_number_state():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 1)
    assert np.allclose(
        state.fock_probabilities,
        [0.0, 0.3454915, 0.6545085, 0.0, 0.0, 0.0],
    )


def test_create_and_annihilate_number_state():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 1)
    assert np.allclose(
        state.fock_probabilities,
        [1, 0, 0, 0, 0, 0],
    )


def test_create_annihilate_and_create():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 1)
    assert np.allclose(
        state.fock_probabilities,
        [0.0, 0.3454915, 0.6545085, 0.0, 0.0, 0.0],
    )


def test_overflow_with_zero_norm_raises_InvalidState_when_normalized():
    with pq.Program() as program:
        pq.Q(2) | pq.StateVector([1]) * np.sqrt(2 / 5)
        pq.Q(1) | pq.StateVector([1]) * np.sqrt(3 / 5)

        pq.Q(1, 2) | pq.Create()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.normalize()

    assert error.value.args[0] == "The norm of the state is 0."


def test_creation_on_multiple_modes():
    with pq.Program() as program:
        pq.Q(2) | pq.StateVector([1]) * np.sqrt(2 / 5)
        pq.Q(1) | pq.StateVector([1]) * np.sqrt(3 / 5)

        pq.Q(1, 2) | pq.Create()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 1)

    assert np.allclose(
        state.fock_probabilities,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3 / 5,
            2 / 5,
            0.0,
        ],
    )


def test_state_normalize_after_overflow():
    with pq.Program() as program:
        pq.Q(2) | pq.StateVector([1]) * np.sqrt(2 / 6)
        pq.Q(1) | pq.StateVector([1]) * np.sqrt(3 / 6)
        pq.Q(2) | pq.StateVector([2]) * np.sqrt(1 / 6)

        pq.Q(2) | pq.Create()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    state.normalize()

    assert np.isclose(state.norm, 1)

    assert np.allclose(
        state.fock_probabilities,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
    )


def test_state_vector_with_fock_amplitude_map_preparation():
    amplitude_map = {(0,): 0.6, (1,): 0.8}

    with pq.Program() as program:
        pq.Q() | pq.StateVector(fock_amplitude_map=amplitude_map)

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=2))

    state = simulator.execute(program).state

    assert np.allclose(state.state_vector, np.array([0.6, 0.8]))


def test_state_vector_with_fock_amplitude_map_and_coefficient():
    amplitude_map = {(0,): 0.6, (1,): 0.8}
    coefficient = 1 / np.sqrt(2)

    with pq.Program() as program_with_mul:
        pq.Q() | pq.StateVector(fock_amplitude_map=amplitude_map) * coefficient

    with pq.Program() as program_with_param:
        pq.Q() | pq.StateVector(
            fock_amplitude_map=amplitude_map, coefficient=coefficient
        )

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=2))

    state_with_mul = simulator.execute(program_with_mul).state
    state_with_param = simulator.execute(program_with_param).state

    expected = coefficient * np.array([0.6, 0.8])

    assert np.allclose(state_with_mul.state_vector, expected)
    assert np.allclose(state_with_param.state_vector, expected)


def test_state_vector_with_fock_amplitude_map_invalid_shape_raises_InvalidState():
    amplitude_map = {(0, 0): 0.6}

    with pq.Program() as program:
        pq.Q() | pq.StateVector(fock_amplitude_map=amplitude_map)

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=2, validate=True))
    with pytest.raises(pq.api.exceptions.InvalidState):
        simulator.execute(program)
