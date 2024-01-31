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


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.SamplingSimulator,
    ),
)
def test_get_particle_detection_probability_raises_PiquassoException_wrong_modes(
    SimulatorClass,
):
    wrong_occupation_number = (0, 1)

    with pq.Program() as program:
        pq.Q(0) | pq.Phaseshifter(np.pi / 3)

    simulator = SimulatorClass(d=3)

    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.PiquassoException):
        state.get_particle_detection_probability(
            occupation_number=wrong_occupation_number
        )
