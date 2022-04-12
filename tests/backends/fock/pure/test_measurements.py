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

import numpy as np

import piquasso as pq


def test_measure_particle_number_on_one_mode():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1, 1]) * np.sqrt(2 / 6)

        pq.Q(2) | pq.StateVector([1]) * np.sqrt(1 / 6)
        pq.Q(2) | pq.StateVector([2]) * np.sqrt(3 / 6)

        pq.Q(2) | pq.ParticleNumberMeasurement()

    simulator = pq.PureFockSimulator(d=3)

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (1,) or sample == (2,)

    if sample == (1,):
        expected_simulator = pq.PureFockSimulator(d=3)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                0.5773502691896258 * pq.StateVector([0, 0, 1]),
                0.816496580927726 * pq.StateVector([0, 1, 1]),
            ]
        ).state

    elif sample == (2,):
        expected_simulator = pq.PureFockSimulator(d=3)
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0, 0, 2])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_two_modes():
    with pq.Program() as program:
        pq.Q(1, 2) | pq.StateVector([1, 1]) * np.sqrt(2 / 6)
        pq.Q(1, 2) | pq.StateVector([0, 1]) * np.sqrt(1 / 6)
        pq.Q(1, 2) | pq.StateVector([0, 2]) * np.sqrt(3 / 6)

        pq.Q(1, 2) | pq.ParticleNumberMeasurement()

    simulator = pq.PureFockSimulator(d=3)

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 1) or sample == (1, 1) or sample == (0, 2)

    if sample == (0, 1):
        expected_simulator = pq.PureFockSimulator(d=3)
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0, 0, 1])]
        ).state

    elif sample == (1, 1):
        expected_simulator = pq.PureFockSimulator(d=3)
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0, 1, 1])]
        ).state

    elif sample == (0, 2):
        expected_simulator = pq.PureFockSimulator(d=3)
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0, 0, 2])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_all_modes():
    config = pq.Config(cutoff=2)

    simulator = pq.PureFockSimulator(d=3, config=config)

    with pq.Program() as program:
        pq.Q() | 0.5 * pq.StateVector([0, 0, 0])
        pq.Q() | 0.5 * pq.StateVector([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.StateVector([1, 0, 0])

        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 0, 0) or sample == (1, 0, 0) or sample == (0, 0, 1)

    if sample == (0, 0, 0):
        expected_simulator = pq.PureFockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.StateVector([0, 0, 0]),
            ],
        ).state

    elif sample == (0, 0, 1):
        expected_simulator = pq.PureFockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.StateVector([0, 0, 1]),
            ]
        ).state

    elif sample == (1, 0, 0):
        expected_simulator = pq.PureFockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.StateVector([1, 0, 0]),
            ],
        ).state

    assert result.state == expected_state


def test_measure_particle_number_with_multiple_shots():
    shots = 4

    # TODO: This is very unusual, that we need to know the cutoff for specifying the
    # state. It should be imposed, that the only parameter for a state should be `d` and
    #  `config` maybe.
    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=2))

    with pq.Program() as program:
        pq.Q() | 0.5 * pq.StateVector([0, 0, 0])
        pq.Q() | 0.5 * pq.StateVector([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.StateVector([1, 0, 0])

        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots)

    assert np.isclose(sum(result.state.fock_probabilities), 1)
    assert len(result.samples) == shots
