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


def test_HomodyneMeasurement_one_mode():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=20, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=1.0)

        pq.Q(0) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            [1.74960435],
            [0.27660972],
            [0.86902385],
            [0.77861923],
            [0.755815],
            [2.04047249],
            [2.42392663],
            [0.99484774],
            [2.06082907],
            [2.28110812],
            [1.43722324],
            [0.92598922],
            [2.07300598],
            [0.85314593],
            [1.87234924],
            [1.64877767],
            [2.4443],
            [0.89617885],
            [2.00713962],
            [1.44643782],
        ],
    )


def test_HomodyneMeasurement_two_modes():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=7, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)

        pq.Q(0, 1) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            [1.04248244, -1.84398138],
            [0.16191613, -1.34935628],
            [0.04870163, -0.14112144],
            [1.7169028, -1.16870184],
            [1.35378134, 0.15965135],
            [0.73006562, -1.16096225],
            [1.36596076, -1.2949428],
            [1.16525672, -0.45997211],
            [1.73727326, -1.26577442],
            [1.30008007, -0.68581022],
            [0.18825746, -1.39577179],
            [0.70315301, -0.51975498],
            [0.0714127, -2.24813514],
            [0.65586351, -0.24068384],
            [1.69412448, -0.52716291],
            [1.68723419, 0.02688775],
            [0.15661024, 0.02666197],
            [1.14206796, -1.12222433],
            [1.29485143, 0.08064955],
            [0.33514897, -0.65509742],
        ],
    )


def test_HomodyneMeasurement_two_modes_with_1_mode_sampled():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=7, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)

        pq.Q(0) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            [1.04248244],
            [-0.43057477],
            [0.16191613],
            [0.07150249],
            [0.04870163],
            [1.33342043],
            [1.7169028],
            [0.28775002],
            [1.35378134],
            [1.57409005],
            [0.73006562],
            [0.21888906],
            [1.36596076],
            [0.14603569],
            [1.16525672],
            [0.94163604],
            [1.73727326],
            [0.1890752],
            [1.30008007],
            [0.73928001],
        ],
    )


def test_HomodyneMeasurement_different_hbar_values():
    shots = 20

    simulator_hbar_2 = pq.PureFockSimulator(
        d=3, config=pq.Config(cutoff=7, seed_sequence=123, hbar=2)
    )
    simulator_hbar_3 = pq.PureFockSimulator(
        d=3, config=pq.Config(cutoff=7, seed_sequence=123, hbar=3)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)
        pq.Q(2) | pq.Squeezing(r=0.1)
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0, 2) | pq.HomodyneMeasurement()

    samples_hbar_2 = simulator_hbar_2.execute(program, shots).samples
    samples_hbar_3 = simulator_hbar_3.execute(program, shots).samples

    assert np.allclose(samples_hbar_2 / np.sqrt(2), samples_hbar_3 / np.sqrt(3))
