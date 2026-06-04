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

from fractions import Fraction

import numpy as np
import pytest

import piquasso as pq


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_ParticleNumberMeasurement_on_all_modes(connector):
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=3, config=pq.Config(cutoff=4), connector=connector
    )

    result = simulator.execute(program, shots=3)

    assert result.samples == [(1, 0, 1)] * 3
    assert result.state is None


@for_all_connectors
def test_ParticleNumberMeasurement_on_one_mode_projects_state(connector):
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1]) / np.sqrt(2)
        pq.Q() | pq.StateVector([0, 1, 0]) / np.sqrt(2)
        pq.Q(0) | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=3, config=pq.Config(cutoff=4, seed_sequence=0), connector=connector
    )

    result = simulator.execute(program, shots=None)

    assert set(result.outcome_map) == {(0,), (1,)}
    assert result.outcome_map[(0,)]["frequency"] == pytest.approx(0.5)
    assert result.outcome_map[(1,)]["frequency"] == pytest.approx(0.5)

    assert np.allclose(
        result.outcome_map[(0,)]["state"].fock_probabilities_map[(1, 0)],
        1.0,
    )
    assert np.allclose(
        result.outcome_map[(1,)]["state"].fock_probabilities_map[(0, 1)],
        1.0,
    )


@for_all_connectors
def test_ParticleNumberMeasurement_shots_none_returns_exact_frequencies(connector):
    with pq.Program() as program:
        pq.Q() | np.sqrt(0.25) * pq.StateVector([0, 0])
        pq.Q() | np.sqrt(0.75) * pq.StateVector([1, 1])
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=connector
    )

    result = simulator.execute(program, shots=None)

    assert result.outcome_map[(0, 0)]["frequency"] == pytest.approx(0.25)
    assert result.outcome_map[(1, 1)]["frequency"] == pytest.approx(0.75)


@for_all_connectors
def test_ParticleNumberMeasurement_with_multiple_shots(connector):
    shots = 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1, 0])
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=3, config=pq.Config(cutoff=4), connector=connector
    )

    result = simulator.execute(program, shots=shots)

    assert result.samples == [(0, 1, 0)] * shots


@for_all_connectors
def test_ParticleNumberMeasurement_after_interferometer(connector):
    shots = 10

    interferometer = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])
        pq.Q() | pq.Interferometer(interferometer)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=3, config=pq.Config(cutoff=4), connector=connector
    )

    result = simulator.execute(program, shots=shots)

    assert len(result.samples) == shots
    assert all(sum(sample) == 2 for sample in result.samples)


@for_all_connectors
def test_ParticleNumberMeasurement_mid_circuit_reindexes_remaining_modes(connector):
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])
        pq.Q(0) | pq.ParticleNumberMeasurement()
        pq.Q(2) | pq.Phaseshifter(phi=np.pi)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=3, config=pq.Config(cutoff=4), connector=connector
    )

    result = simulator.execute(program, shots=1)

    assert result.samples == [(1, 0, 1)]
    assert result.branches[0].frequency == Fraction(1)
