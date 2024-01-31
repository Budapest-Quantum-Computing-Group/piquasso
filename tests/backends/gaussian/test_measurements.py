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


@pytest.fixture
def d():
    return 3


@pytest.fixture
def state(d):
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=2, phi=np.pi / 3)
        pq.Q(1) | pq.Displacement(r=1, phi=np.pi / 4)
        pq.Q(2) | pq.Displacement(r=1 / 2, phi=np.pi / 6)

        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi / 4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    return state


@pytest.fixture
def nondisplaced_state(d):
    with pq.Program() as program:
        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi / 4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    return state


def test_measure_homodyne_zeroes_state_on_measured_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    assert state.xpxp_mean_vector[0] == 0
    assert state.xpxp_covariance_matrix[0][0] == state._config.hbar


def test_measure_homodyne_with_angle_does_not_alter_the_state(state):
    angle = np.pi / 3

    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    measured_state = simulator.execute(program, initial_state=state).state
    measured_state.validate()

    with pq.Program() as program_with_rotation:
        pq.Q(0) | pq.HomodyneMeasurement(angle)

    state_with_rotation = simulator.execute(
        program_with_rotation, initial_state=state
    ).state
    state_with_rotation.validate()

    assert measured_state == state_with_rotation


def test_measure_homodyne_with_multiple_shots(state):
    shots = 4

    with pq.Program() as program:
        pq.Q(0, 1) | pq.HomodyneMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state, shots=shots)

    assert len(result.samples) == shots


def test_measure_heterodyne_zeroes_state_on_measured_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HeterodyneMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    assert state.xpxp_mean_vector[0] == 0
    assert state.xpxp_covariance_matrix[0][0] == state._config.hbar


def test_measure_heterodyne_with_multiple_shots(state):
    shots = 4

    with pq.Program() as program:
        pq.Q(0, 1) | pq.HeterodyneMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state, shots=shots)

    assert len(result.samples) == shots


def test_measure_dyne(state):
    detection_covariance = np.array(
        [
            [2, 0],
            [0, 0.5],
        ]
    )

    with pq.Program() as program:
        pq.Q(0) | pq.GeneraldyneMeasurement(detection_covariance)

    simulator = pq.GaussianSimulator(d=state.d)
    evolved_state = simulator.execute(program, initial_state=state).state

    evolved_state.validate()


def test_measure_dyne_with_multiple_shots(state):
    shots = 4

    detection_covariance = np.array(
        [
            [2, 0],
            [0, 0.5],
        ]
    )

    with pq.Program() as program:
        pq.Q(0) | pq.GeneraldyneMeasurement(detection_covariance)

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state, shots=shots)

    assert len(result.samples) == shots


def test_measure_particle_number_on_one_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state)

    assert result


def test_measure_particle_number_on_two_modes(state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state)

    assert result


def test_measure_particle_number_on_all_modes(state):
    with pq.Program() as program:
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)
    result = simulator.execute(program, initial_state=state)

    assert result


@pytest.mark.parametrize(
    "MeasurementClass",
    (pq.HomodyneMeasurement, pq.HeterodyneMeasurement),
)
def test_dyneMeasurement_resulting_state_inherits_config(MeasurementClass):
    with pq.Program() as program:
        pq.Q(0) | pq.Fourier()
        pq.Q(0, 2) | pq.Squeezing2(r=0.5, phi=0)
        pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0)
        pq.Q(0) | MeasurementClass()

    custom_config = pq.Config(hbar=3, seed_sequence=42)

    simulator = pq.GaussianSimulator(d=3, config=custom_config)

    state = simulator.execute(program).state

    assert state._config == custom_config


@pytest.mark.parametrize(
    "MeasurementClass",
    (pq.HomodyneMeasurement, pq.HeterodyneMeasurement),
)
def test_dyneMeasurement_results_in_same_state_regardless_of_hbar(MeasurementClass):
    seed_sequence = 42

    with pq.Program() as program:
        pq.Q(0) | pq.Fourier()
        pq.Q(0, 2) | pq.Squeezing2(r=0.5, phi=0)
        pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0)
        pq.Q(0) | MeasurementClass()

    simulator_hbar_1 = pq.GaussianSimulator(
        d=3, config=pq.Config(hbar=1, seed_sequence=seed_sequence)
    )
    simulator_hbar_3 = pq.GaussianSimulator(
        d=3, config=pq.Config(hbar=3, seed_sequence=seed_sequence)
    )

    state_hbar_1 = simulator_hbar_1.execute(program).state
    state_hbar_3 = simulator_hbar_3.execute(program).state

    assert state_hbar_1 == state_hbar_3


def test_displaced_ThresholdMeasurement_raises_NotImplementedError_with_torontonian(
    state,
):
    state._config.use_torontonian = True

    state_with_nonzero_displacements = state

    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)

    with pytest.raises(NotImplementedError):
        simulator.execute(program, initial_state=state_with_nonzero_displacements)


def test_measure_threshold_on_one_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=nondisplaced_state.d)
    result = simulator.execute(program, initial_state=nondisplaced_state)

    assert result


def test_measure_threshold_with_multiple_shots(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=nondisplaced_state.d)
    result = simulator.execute(program, initial_state=nondisplaced_state, shots=20)

    assert result


def test_measure_threshold_on_two_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=nondisplaced_state.d)
    result = simulator.execute(program, initial_state=nondisplaced_state)

    assert result


def test_measure_threshold_on_all_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q() | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=nondisplaced_state.d)
    result = simulator.execute(program, initial_state=nondisplaced_state)

    assert result
