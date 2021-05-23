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
def program(d):
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=d)

        pq.Q(0) | pq.Displacement(r=2, phi=np.pi/3)
        pq.Q(1) | pq.Displacement(r=1, phi=np.pi/4)
        pq.Q(2) | pq.Displacement(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi/4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    return program


@pytest.fixture
def nondisplaced_program(d):
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=d)

        pq.Q(1) | pq.Squeezing(r=1 / 2, phi=np.pi/4)
        pq.Q(2) | pq.Squeezing(r=3 / 4)

    return program


def test_measure_homodyne(program):
    with program:
        pq.Q(0) | pq.HomodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, )

    program.state.validate()


def test_measure_homodyne_zeroes_state_on_measured_modes(program):
    with program:
        pq.Q(0) | pq.HomodyneMeasurement()

    program.execute()
    program.state.validate()

    assert program.state.mean[0] == 0
    assert program.state.cov[0][0] == pq.constants.HBAR


def test_measure_homodyne_with_rotation(program):
    angle = np.pi / 3

    with program:
        pq.Q(0) | pq.HomodyneMeasurement(angle)

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, )


def test_measure_homodyne_with_angle_does_not_alter_the_state(program):
    angle = np.pi / 3

    program_with_rotation = program.copy()

    with program:
        pq.Q(0) | pq.HomodyneMeasurement()

    program.execute()
    program.state.validate()

    with program_with_rotation:
        pq.Q(0) | pq.HomodyneMeasurement(angle)

    program_with_rotation.execute()
    program_with_rotation.state.validate()

    assert program.state == program_with_rotation.state


def test_measure_homodyne_on_multiple_modes(program):
    with program:
        pq.Q(0, 1) | pq.HomodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, 1)

    program.state.validate()


def test_measure_homodyne_on_all_modes(program, d):
    with program:
        pq.Q() | pq.HomodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == tuple(range(d))

    program.state.validate()


def test_measure_homodyne_with_multiple_shots(program):
    shots = 4

    with program:
        pq.Q(0, 1) | pq.HomodyneMeasurement(shots=shots)

    results = program.execute()

    assert len(results[0].samples) == shots


def test_measure_heterodyne(program):
    with program:
        pq.Q(0) | pq.HeterodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, )

    program.state.validate()


def test_measure_heterodyne_zeroes_state_on_measured_modes(program):
    with program:
        pq.Q(0) | pq.HeterodyneMeasurement()

    program.execute()
    program.state.validate()

    assert program.state.mean[0] == 0
    assert program.state.cov[0][0] == pq.constants.HBAR


def test_measure_heterodyne_on_multiple_modes(program):
    with program:
        pq.Q(0, 1) | pq.HeterodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, 1)

    program.state.validate()


def test_measure_heterodyne_on_all_modes(program, d):
    with program:
        pq.Q() | pq.HeterodyneMeasurement()

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == tuple(range(d))

    program.state.validate()


def test_measure_heterodyne_with_multiple_shots(program):
    shots = 4

    with program:
        pq.Q(0, 1) | pq.HeterodyneMeasurement(shots=shots)

    results = program.execute()

    assert len(results[0].samples) == shots


def test_measure_dyne(program):
    detection_covariance = np.array(
        [
            [2, 0],
            [0, 0.5],
        ]
    )

    with program:
        pq.Q(0) | pq.GeneraldyneMeasurement(detection_covariance)

    results = program.execute()

    assert len(results) == 1
    assert results[0].instruction.modes == (0, )

    program.state.validate()


def test_measure_dyne_with_multiple_shots(program):
    shots = 4

    detection_covariance = np.array(
        [
            [2, 0],
            [0, 0.5],
        ]
    )

    with program:
        pq.Q(0) | pq.GeneraldyneMeasurement(detection_covariance, shots=shots)

    results = program.execute()

    assert len(results[0].samples) == shots


def test_measure_particle_number_on_one_modes(program):
    with program:
        pq.Q(0) | pq.ParticleNumberMeasurement(cutoff=4)

    results = program.execute()

    assert results


def test_measure_particle_number_on_two_modes(program):
    with program:
        pq.Q(0, 1) | pq.ParticleNumberMeasurement(cutoff=4)

    results = program.execute()

    assert results


def test_measure_particle_number_on_all_modes(program):
    with program:
        pq.Q() | pq.ParticleNumberMeasurement(cutoff=4)

    results = program.execute()

    assert results


def test_measure_threshold_raises_NotImplementedError_for_nonzero_displacements(
    program,
):
    program_with_nonzero_displacements = program

    with program_with_nonzero_displacements:
        pq.Q(0) | pq.ThresholdMeasurement()

    with pytest.raises(NotImplementedError):
        program_with_nonzero_displacements.execute()


def test_measure_threshold_on_one_modes(nondisplaced_program):
    with nondisplaced_program:
        pq.Q(0) | pq.ThresholdMeasurement()

    results = nondisplaced_program.execute()

    assert results


def test_measure_threshold_with_multiple_shots(nondisplaced_program):
    with nondisplaced_program:
        pq.Q(0) | pq.ThresholdMeasurement(shots=20)

    results = nondisplaced_program.execute()

    assert results


def test_measure_threshold_on_two_modes(nondisplaced_program):
    with nondisplaced_program:
        pq.Q(0, 1) | pq.ThresholdMeasurement()

    results = nondisplaced_program.execute()

    assert results


def test_measure_threshold_on_all_modes(nondisplaced_program):
    with nondisplaced_program:
        pq.Q() | pq.ThresholdMeasurement()

    results = nondisplaced_program.execute()

    assert results


def test_multiple_particle_number_measurements_in_one_program():
    first_measurement_shots = 3
    second_measurement_shots = 4

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | pq.Squeezing(r=1, phi=0)
        pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter(theta=1, phi=np.pi / 4)
        pq.Q(0, 1) | pq.ParticleNumberMeasurement(cutoff=5, shots=3)
        pq.Q(2) | pq.ParticleNumberMeasurement(cutoff=5, shots=4)

    results = program.execute()

    assert len(results) == 2

    assert len(results[0].samples) == first_measurement_shots
    assert len(results[1].samples) == second_measurement_shots
