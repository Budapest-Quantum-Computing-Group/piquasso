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
import pytest

import piquasso as pq
from piquasso.api.exceptions import InvalidParameter


@pytest.fixture
def interferometer_matrix():
    return np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
    )


def test_sampling_raises_InvalidParameter_for_negative_shot_value(
    interferometer_matrix,
):
    invalid_shots = -1

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(InvalidParameter):
        simulator.execute(program, invalid_shots)


def test_sampling_samples_number(interferometer_matrix):
    shots = 100

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    assert len(result.samples) == shots, (
        f"Expected {shots} samples, " f"got: {len(program.result)}"
    )


def test_sampling_mode_permutation(interferometer_matrix):
    shots = 1

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    sample = result.samples[0]
    assert np.allclose(
        sample, [1, 0, 0, 1, 1]
    ), f"Expected [1, 0, 0, 1, 1], got: {sample}"


def test_sampling_multiple_samples_for_permutation_interferometer(
    interferometer_matrix,
):
    shots = 2

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    samples = result.samples
    first_sample = samples[0]
    second_sample = samples[1]

    assert np.allclose(
        first_sample, second_sample
    ), f"Expected same samples, got: {first_sample} & {second_sample}"


def test_mach_zehnder():
    int_ = np.pi / 3
    ext = np.pi / 4

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q(0, 1) | pq.MachZehnder(int_=int_, ext=ext)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)


def test_fourier():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q(0) | pq.Fourier()
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)


def test_uniform_loss():
    d = 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        for i in range(d):
            pq.Q(i) | pq.Loss(transmissivity=0.9)

        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=d)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy


def test_general_loss():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0) | pq.Loss(transmissivity=0.4)
        pq.Q(1) | pq.Loss(transmissivity=0.5)

        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)
