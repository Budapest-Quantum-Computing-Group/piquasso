#
# Copyright 2021 Budapest Quantum Computing Group
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
        pq.Q(0) | pq.Displacement(r=2, phi=np.pi/3)
        pq.Q(1) | pq.Displacement(r=1, phi=np.pi/4)
        pq.Q(2) | pq.Displacement(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi/4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    state = pq.GaussianState(d=d)
    state.apply(program)

    return state


@pytest.fixture
def nondisplaced_state(d):
    with pq.Program() as program:
        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi/4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    state = pq.GaussianState(d=d)
    state.apply(program)

    return state


def test_measure_homodyne(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == (0, )

    state.validate()


def test_measure_homodyne_zeroes_state_on_measured_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement()

    state.apply(program)
    state.validate()

    assert state.mean[0] == 0
    assert state.cov[0][0] == pq.constants.HBAR


def test_measure_homodyne_with_rotation(state):
    angle = np.pi / 3

    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement(angle)

    result = state.apply(program)

    assert result.instruction.modes == (0, )


def test_measure_homodyne_with_angle_does_not_alter_the_state(state):
    angle = np.pi / 3

    state_with_rotation = state.copy()

    with pq.Program() as program:
        pq.Q(0) | pq.HomodyneMeasurement()

    state.apply(program)
    state.validate()

    with pq.Program() as program_with_rotation:
        pq.Q(0) | pq.HomodyneMeasurement(angle)

    state_with_rotation.apply(program_with_rotation)
    state_with_rotation.validate()

    assert state == state_with_rotation


def test_measure_homodyne_on_multiple_modes(state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.HomodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == (0, 1)

    state.validate()


def test_measure_homodyne_on_all_modes(state, d):
    with pq.Program() as program:
        pq.Q() | pq.HomodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == tuple(range(d))

    state.validate()


def test_measure_homodyne_with_multiple_shots(state):
    shots = 4

    with pq.Program() as program:
        pq.Q(0, 1) | pq.HomodyneMeasurement()

    result = state.apply(program, shots=shots)

    assert len(result.samples) == shots


def test_measure_heterodyne(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HeterodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == (0, )

    state.validate()


def test_measure_heterodyne_zeroes_state_on_measured_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.HeterodyneMeasurement()

    state.apply(program)
    state.validate()

    assert state.mean[0] == 0
    assert state.cov[0][0] == pq.constants.HBAR


def test_measure_heterodyne_on_multiple_modes(state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.HeterodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == (0, 1)

    state.validate()


def test_measure_heterodyne_on_all_modes(state, d):
    with pq.Program() as program:
        pq.Q() | pq.HeterodyneMeasurement()

    result = state.apply(program)

    assert result.instruction.modes == tuple(range(d))

    state.validate()


def test_measure_heterodyne_with_multiple_shots(state):
    shots = 4

    with pq.Program() as program:
        pq.Q(0, 1) | pq.HeterodyneMeasurement()

    result = state.apply(program, shots=shots)

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

    result = state.apply(program)

    assert result.instruction.modes == (0, )

    state.validate()


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

    result = state.apply(program, shots=shots)

    assert len(result.samples) == shots


def test_measure_particle_number_on_one_modes(state):
    with pq.Program() as program:
        pq.Q(0) | pq.ParticleNumberMeasurement(cutoff=4)

    result = state.apply(program)

    assert result


def test_measure_particle_number_on_two_modes(state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.ParticleNumberMeasurement(cutoff=4)

    result = state.apply(program)

    assert result


def test_measure_particle_number_on_all_modes(state):
    with pq.Program() as program:
        pq.Q() | pq.ParticleNumberMeasurement(cutoff=4)

    result = state.apply(program)

    assert result


def test_measure_threshold_raises_NotImplementedError_for_nonzero_displacements(
    state,
):
    state_with_nonzero_displacements = state

    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    with pytest.raises(NotImplementedError):
        state_with_nonzero_displacements.apply(program)


def test_measure_threshold_on_one_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    result = nondisplaced_state.apply(program)

    assert result


def test_measure_threshold_with_multiple_shots(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    result = nondisplaced_state.apply(program, shots=20)

    assert result


def test_measure_threshold_on_two_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.ThresholdMeasurement()

    result = nondisplaced_state.apply(program)

    assert result


def test_measure_threshold_on_all_modes(nondisplaced_state):
    with pq.Program() as program:
        pq.Q() | pq.ThresholdMeasurement()

    result = nondisplaced_state.apply(program)

    assert result
