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

import piquasso as pq


def test_Loss_uniform():
    d = 5
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        for i in range(5):
            pq.Q(i) | pq.Loss(0.9)

    simulator = pq.SamplingSimulator(d=d)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy

    singular_values = np.linalg.svd(state.interferometer)[1]

    assert np.allclose(singular_values, [0.9, 0.9, 0.9, 0.9, 0.9])


def test_Loss_non_uniform():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0) | pq.Loss(transmissivity=0.4)
        pq.Q(1) | pq.Loss(transmissivity=0.5)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy

    singular_values = np.linalg.svd(state.interferometer)[1]

    assert len(singular_values[np.isclose(singular_values, 0.4)]) == 1
    assert len(singular_values[np.isclose(singular_values, 0.5)]) == 1
    assert len(singular_values[np.isclose(singular_values, 1.0)]) == 3
