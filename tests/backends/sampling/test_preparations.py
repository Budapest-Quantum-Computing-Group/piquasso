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


def test_initial_state():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_initial_state = [1, 1, 1, 0, 0]
    assert np.allclose(state.initial_state, expected_initial_state)


def test_initial_state_multiplied_with_coefficient():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 2.0

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_initial_state = [2, 2, 2, 0, 0]
    assert np.allclose(state.initial_state, expected_initial_state)


def test_initial_state_raises_InvalidState_for_noninteger_input_state():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 0.5

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(pq.api.exceptions.InvalidState):
        simulator.execute(program)


def test_initial_state_raises_InvalidState_when_multiple_StateVectors_specified():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.StateVector([1, 2, 0, 3, 0])

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(pq.api.exceptions.InvalidState):
        simulator.execute(program)


def test_interferometer_init():
    state = pq.SamplingState(d=5)

    expected_interferometer = np.diag(np.ones(state.d, dtype=complex))
    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_neighbouring_modes():
    U = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1, 2) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_gaped_modes():
    U = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1, 4) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 2, 0, 0, 3],
            [4, 5, 0, 0, 6],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [7, 8, 0, 0, 9],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_reversed_gaped_modes():
    U = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(4, 3, 1) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_probability_distribution():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

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
            0.6945423895038292,
            0.30545762086020883,
            0.0,
        ],
    )


def test_get_particle_detection_probability():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability(occupation_number=(1, 1, 0))

    assert np.allclose(probability, 0.30545762086020883)


def test_get_particle_detection_probability_on_different_subspace():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

    different_particle_subspace_occupation_number = (3, 1, 0)

    probability = state.get_particle_detection_probability(
        occupation_number=different_particle_subspace_occupation_number
    )

    assert np.allclose(probability, 0.0)
