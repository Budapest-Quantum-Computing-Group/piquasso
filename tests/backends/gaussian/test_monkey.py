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
import random
import numpy as np

from scipy.stats import unitary_group

import piquasso as pq


@pytest.mark.monkey
@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_random_gaussianstate(d):
    with pq.Program() as prepare_random_program:
        for i in range(d):
            r = np.random.normal()
            phi = random.uniform(0, 2 * np.pi)

            pq.Q(i) | pq.Squeezing(r=r, phi=phi)

        random_unitary = np.array(unitary_group.rvs(d))

        pq.Q(all) | pq.Interferometer(random_unitary)

    with pq.Program() as program:
        pq.Q(all) | prepare_random_program

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state
    state.validate()
