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

from scipy.stats import unitary_group


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


def test_post_select_NS_gate():
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 0.25)

    assert np.isclose(state.norm, 0.25)

    state.normalize()
    state.validate()

    purity = state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_state_vector = np.copy(first_mode_state_vector)
    expected_state_vector[2] *= -1

    assert np.allclose(
        state.density_matrix[:d, :d],
        np.outer(expected_state_vector, expected_state_vector),
    )


def test_post_select_random_unitary():
    d = 3

    interferometer_matrix = unitary_group.rvs(d)

    postselect_modes = (1, 2)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
        pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
        pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=postselect_modes, photon_counts=(1, 0)
        )

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    state.normalize()
    state.validate()

    assert state.d == d - len(postselect_modes)


def test_post_select_conditional_sign_flip_gate_with_1_over_16_success_rate():
    ancilla_modes = (4, 5, 6, 7)

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as nonlinear_shift:
        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

    with pq.Program() as conditional_sign_flip:
        pq.Q(0, 2) | pq.Beamsplitter(theta=np.pi / 4)

        pq.Q(0, ancilla_modes[0], ancilla_modes[1]) | nonlinear_shift
        pq.Q(2, ancilla_modes[2], ancilla_modes[3]) | nonlinear_shift

        pq.Q(0, 2) | pq.Beamsplitter(theta=-np.pi / 4)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    states = [state_00, state_01, state_10, state_11]
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    ancilla_state = [1, 0]

    with pq.Program() as program:
        for input_state, coeff in zip(states, coefficients):
            pq.Q(all) | pq.StateVector(input_state + ancilla_state * 2) * coeff

        pq.Q(all) | conditional_sign_flip

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=ancilla_modes, photon_counts=ancilla_state * 2
        )

    simulator = pq.PureFockSimulator(d=8, config=pq.Config(cutoff=5))

    final_state = simulator.execute(program).state

    expected_success_rate = 1 / 16
    actual_success_rate = final_state.norm

    assert np.isclose(expected_success_rate, actual_success_rate)

    final_state.normalize()

    purity = final_state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    expected_coefficients[3] *= -1

    with pq.Program() as expectation_program:
        for input_state, coeff in zip(states, expected_coefficients):
            pq.Q(all) | pq.StateVector(input_state) * coeff

    expectation_simulator = pq.PureFockSimulator(d=4, config=pq.Config(cutoff=5))
    expected_state = expectation_simulator.execute(expectation_program).state

    assert np.allclose(expected_state.density_matrix, final_state.density_matrix)
