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


def test_displaced_ThresholdMeasurement_with_torontonian(state):
    state._config.use_torontonian = True

    state_with_nonzero_displacements = state

    with pq.Program() as program:
        pq.Q(0) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(d=state.d)

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


def test_seeded_gaussian_boson_sampling():
    d = 5
    shots = 10

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as gaussian_boson_sampling:
        pq.Q(all) | pq.Graph(A)

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator1 = pq.GaussianSimulator(
        d=d, connector=pq.NumpyConnector(), config=pq.Config(seed_sequence=123)
    )
    result1 = simulator1.execute(gaussian_boson_sampling, shots=shots)

    simulator2 = pq.GaussianSimulator(
        d=d, connector=pq.NumpyConnector(), config=pq.Config(seed_sequence=123)
    )
    result2 = simulator2.execute(gaussian_boson_sampling, shots=shots)

    assert np.allclose(result1.samples, result2.samples)


def test_seeded_gaussian_boson_sampling_samples_from_graph():
    d = 5
    shots = 50

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as gaussian_boson_sampling:
        pq.Q(all) | pq.Graph(A)

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=d, connector=pq.NumpyConnector(), config=pq.Config(seed_sequence=123)
    )
    samples = simulator.execute(gaussian_boson_sampling, shots=shots).samples

    assert np.allclose(
        samples,
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [2, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 2, 1, 4, 3],
            [0, 0, 0, 0, 0],
            [4, 4, 0, 2, 4],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [2, 0, 0, 1, 1],
            [3, 1, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [3, 1, 2, 4, 2],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 2, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [2, 0, 1, 3, 2],
            [2, 1, 0, 0, 1],
            [1, 4, 0, 0, 3],
            [3, 4, 1, 3, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 4, 3],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 2, 2],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [2, 3, 0, 2, 3],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 2, 1],
        ],
    )


def test_seeded_gaussian_boson_sampling_samples_displaced():
    d = 5
    shots = 50

    with pq.Program() as gaussian_boson_sampling:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        pq.Q(0, 1, 2) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=d, connector=pq.NumpyConnector(), config=pq.Config(seed_sequence=123)
    )

    samples = simulator.execute(gaussian_boson_sampling, shots=shots).samples

    assert np.allclose(
        samples,
        [
            [1, 1, 1],
            [1, 4, 1],
            [0, 3, 0],
            [0, 1, 1],
            [0, 2, 1],
            [1, 3, 1],
            [0, 4, 0],
            [0, 3, 1],
            [0, 4, 0],
            [0, 3, 0],
            [0, 3, 0],
            [0, 2, 0],
            [0, 0, 0],
            [0, 2, 1],
            [0, 1, 0],
            [2, 0, 0],
            [1, 1, 2],
            [0, 0, 1],
            [0, 0, 2],
            [0, 2, 0],
            [2, 4, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
            [0, 4, 1],
            [0, 4, 0],
            [1, 3, 2],
            [1, 2, 0],
            [1, 4, 1],
            [1, 1, 2],
            [0, 4, 1],
            [0, 2, 0],
            [2, 1, 2],
            [1, 1, 2],
            [0, 2, 1],
            [1, 3, 3],
            [0, 3, 2],
            [0, 2, 0],
            [0, 4, 0],
            [0, 3, 0],
            [0, 2, 0],
            [0, 2, 1],
            [1, 2, 1],
            [0, 3, 1],
            [1, 3, 0],
            [0, 3, 0],
            [0, 2, 2],
            [0, 4, 0],
            [1, 2, 0],
            [0, 1, 1],
        ],
    )


def test_ThresholdMeasurement_use_torontonian_seeding():
    d = 5
    shots = 10

    seed_sequence = 123

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Graph(A)

        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(seed_sequence=seed_sequence, use_torontonian=True),
    )
    result = simulator.execute(program, shots=shots)

    assert result.samples == [
        (1, 0, 0, 0, 1),
        (1, 1, 0, 1, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    ]


def test_ThresholdMeasurement_use_torontonian_seeding_float32():
    d = 5
    shots = 10

    seed_sequence = 123

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Graph(A)

        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(
            seed_sequence=seed_sequence, use_torontonian=True, dtype=np.float32
        ),
    )
    result = simulator.execute(program, shots=shots)

    assert result.samples == [
        (1, 0, 0, 0, 1),
        (1, 1, 0, 1, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    ]
