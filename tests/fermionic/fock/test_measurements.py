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

import numpy as np

import piquasso as pq


def test_ParticleNumberMeasurement_shots_None_projects_remaining_modes():
    probability_100 = 0.25
    probability_011 = 0.75

    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 0, 0]) * np.sqrt(probability_100)
        pq.Q() | pq.NumberState([0, 1, 1]) * np.sqrt(probability_011)

        pq.Q(0) | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(d=3)

    result = simulator.execute(program, shots=None)

    outcome_map = result.outcome_map

    assert set(outcome_map) == {(0,), (1,)}

    assert np.isclose(outcome_map[(1,)]["frequency"], probability_100)
    assert np.isclose(outcome_map[(0,)]["frequency"], probability_011)

    assert np.isclose(
        outcome_map[(1,)]["state"].get_particle_detection_probability([0, 0]),
        1.0,
    )
    assert np.isclose(
        outcome_map[(0,)]["state"].get_particle_detection_probability([1, 1]),
        1.0,
    )


def test_ParticleNumberMeasurement_finite_shots_projects_remaining_modes():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 0, 0])

        pq.Q(0) | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(d=3)

    result = simulator.execute(program, shots=3)

    assert result.get_counts() == {(1,): 3}

    branch = result.branches[0]

    assert branch.outcome == (1,)
    assert np.isclose(
        branch.state.get_particle_detection_probability([0, 0]),
        1.0,
    )


def test_ParticleNumberMeasurement_all_modes_seeded_sampling():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 0, 0]) / 2
        pq.Q() | pq.NumberState([0, 1, 1]) * np.sqrt(3) / 2

        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(d=3, config=pq.Config(seed_sequence=123))

    result = simulator.execute(program, shots=8)

    assert result.get_counts() == {
        (1, 0, 0): 4,
        (0, 1, 1): 4,
    }
    assert all(branch.state is None for branch in result.branches)
