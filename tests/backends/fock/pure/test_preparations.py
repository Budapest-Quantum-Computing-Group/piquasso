#
# Copyright 2021-2023 Budapest Quantum Computing Group
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


def test_overflow_with_zero_norm_raises_InvalidState():
    with pq.Program() as program:
        pq.Q(2) | pq.StateVector([1]) * np.sqrt(2 / 5)
        pq.Q(1) | pq.StateVector([1]) * np.sqrt(3 / 5)

        pq.Q(1, 2) | pq.Create()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=3))

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        simulator.execute(program).state

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


def test_state_is_renormalized_after_overflow():
    with pq.Program() as program:
        pq.Q(2) | pq.StateVector([1]) * np.sqrt(2 / 6)
        pq.Q(1) | pq.StateVector([1]) * np.sqrt(3 / 6)
        pq.Q(2) | pq.StateVector([2]) * np.sqrt(1 / 6)

        pq.Q(2) | pq.Create()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 1)

    assert np.allclose(
        state.fock_probabilities,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
    )
