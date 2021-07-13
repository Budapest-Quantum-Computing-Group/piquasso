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


@pytest.mark.filterwarnings("ignore:.*may not result in the desired state.*")
@pytest.mark.parametrize(
    "StateClass",
    (pq.FockState, pq.PNCFockState, pq.PureFockState)
)
def test_FockState_get_particle_detection_probability(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=4) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    program.execute()

    probability = program.state.get_particle_detection_probability(
        occupation_number=(0, 2)
    )

    assert np.isclose(probability, 0.0012355767142126952)
