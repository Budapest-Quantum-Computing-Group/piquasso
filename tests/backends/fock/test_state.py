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


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_get_particle_detection_probability(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability(occupation_number=(0, 2))

    assert np.isclose(probability, 0.0012355767142126952)


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_quadratures_mean_variance(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 - 0.5j)
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state

    mean, var = state.quadratures_mean_variance(modes=(0,))

    assert np.isclose(mean, 1.6374076, atol=1e-5)
    assert np.isclose(var, 0.6705157, atol=1e-4)
