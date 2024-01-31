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
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (-3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (-1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (-2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(2) | pq.ParticleNumberMeasurement()

    config = pq.Config(cutoff=3)

    simulator = pq.FockSimulator(d=3, config=config)

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (1,) or sample == (2,)

    if sample == (1,):
        expected_simulator = pq.FockSimulator(
            d=3,
            config=config,
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                1 / 3 * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                4j * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)),
                -2j * pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)),
                -4j * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)),
                2 / 3 * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
                2j * pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)),
            ]
        ).state

    elif sample == (2,):
        expected_simulator = pq.FockSimulator(
            d=3,
            config=config,
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_two_modes():
    with pq.Program() as program:
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (-3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (-1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (-2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(1, 2) | pq.ParticleNumberMeasurement()

    config = pq.Config(cutoff=3)

    simulator = pq.FockSimulator(d=3, config=config)

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 1) or sample == (1, 1) or sample == (0, 2)

    if sample == (0, 1):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (-6j),
                pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 6j,
            ]
        ).state

    elif sample == (1, 1):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
            ]
        ).state

    elif sample == (0, 2):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_all_modes():
    with pq.Program() as preparation:
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1 / 8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.ParticleNumberMeasurement()

    config = pq.Config(cutoff=3)

    simulator = pq.FockSimulator(d=3, config=config)

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 0, 0) or sample == (0, 0, 1) or sample == (1, 0, 0)

    if sample == (0, 0, 0):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)),
            ]
        ).state

    elif sample == (0, 0, 1):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
            ]
        ).state

    elif sample == (1, 0, 0):
        expected_simulator = pq.FockSimulator(d=3, config=config)
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)),
            ]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_with_multiple_shots():
    shots = 4

    with pq.Program() as preparation:
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1 / 8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.FockSimulator(d=3, config=pq.Config(cutoff=3))

    result = simulator.execute(program, shots)

    assert np.isclose(sum(result.state.fock_probabilities), 1)
    assert len(result.samples) == shots
