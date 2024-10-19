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


import piquasso as pq

from piquasso._math.combinatorics import partitions

import numpy as np


if __name__ == "__main__":
    unitary = np.array(
        [
            [
                0.90939407 + 0.26443525j,
                0.00450261 + 0.01880791j,
                0.31670376 + 0.0490014j,
            ],
            [
                0.19639913 + 0.19652978j,
                0.23847303 + 0.39887995j,
                -0.67825584 - 0.49678752j,
            ],
            [
                -0.10943493 - 0.11791454j,
                0.35197974 + 0.81225714j,
                0.3122759 + 0.30488117j,
            ],
        ]
    )

    input_state = np.array([3, 1, 1], dtype=int)

    loss_transmittance = 1.0

    n = sum(input_state)
    d = len(input_state)

    N = 10000

    program = pq.Program(
        instructions=[
            pq.StateVector(input_state),
            pq.Interferometer(unitary),
            *[pq.Loss(loss_transmittance).on_modes(i) for i in range(d)],
            pq.ParticleNumberMeasurement(),
        ]
    )

    simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(seed_sequence=42))

    samples = [tuple(sample) for sample in simulator.execute(program, shots=N).samples]

    for partition in partitions(d, n):
        print(partition, samples.count(tuple(partition)))
