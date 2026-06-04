#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from collections import Counter

import numpy as np
import pytest

import piquasso as pq


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


def test_GaussianSimulator_ParticleNumberMeasurement_issue_reproduction(
    generate_unitary_matrix,
):
    d = 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(generate_unitary_matrix(d))
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(d=d, config=pq.Config(seed_sequence=321))

    result = simulator.execute(program, shots=50)

    assert len(result.samples) == 50
    assert all(len(sample) == d for sample in result.samples)
    assert all(set(sample) <= {0, 1} for sample in result.samples)
    assert all(sum(sample) == 3 for sample in result.samples)


@for_all_connectors
def test_GaussianSimulator_ParticleNumberMeasurement_respects_measured_modes(
    connector,
):
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])
        pq.Q(2, 0) | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=3, connector=connector, config=pq.Config(seed_sequence=123)
    )

    result = simulator.execute(program, shots=10)

    assert result.samples == [(1, 1)] * 10


@for_all_connectors
def test_GaussianSimulator_ParticleNumberMeasurement_matches_probabilities(
    connector,
):
    theta = np.pi / 5
    interferometer = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    with pq.Program() as state_program:
        pq.Q() | pq.StateVector([1, 0])
        pq.Q() | pq.Interferometer(interferometer)

    with pq.Program() as measurement_program:
        pq.Q() | pq.StateVector([1, 0])
        pq.Q() | pq.Interferometer(interferometer)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=2, connector=connector, config=pq.Config(seed_sequence=123)
    )

    expected_state = simulator.execute(state_program).state
    expected_probability = expected_state.get_particle_detection_probability([1, 0])

    shots = 2000
    result = simulator.execute(measurement_program, shots=shots)
    counts = Counter(result.samples)

    assert set(counts) <= {(1, 0), (0, 1)}
    assert np.isclose(counts[(1, 0)] / shots, expected_probability, atol=0.05)
