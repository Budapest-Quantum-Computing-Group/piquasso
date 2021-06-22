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

import subprocess

import numpy as np
import matplotlib.pyplot as plt

import piquasso as pq

import strawberryfields as sf


d = 5
shots = 1


adjacency_matrix = np.array(
    [
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0],
    ]
)


with pq.Program() as pq_program:
    pq.Q() | pq.GaussianState(d=d)

    pq.Q() | pq.Graph(adjacency_matrix)

    # NOTE: In SF the cutoff is 5, and couldn't be changed
    pq.Q(0, 1, 2) | pq.ParticleNumberMeasurement(cutoff=5)


sf_program = sf.Program(d)
sf_engine = sf.Engine(backend="gaussian")

with sf_program.context as q:
    sf.ops.GraphEmbed(
        adjacency_matrix,
    ) | tuple([q[i] for i in range(d)])

    sf.ops.MeasureFock() | (q[0], q[1], q[2])


if __name__ == "__main__":
    pq_results = np.array(pq_program.execute(shots=shots)[0].samples)
    sf_results = sf_engine.run(sf_program, shots=shots).samples

    N_points = 100000
    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(pq_results, bins=n_bins)
    axs[1].hist(sf_results, bins=n_bins)

    fig.savefig("histogram.png")

    subprocess.call(('xdg-open', "histogram.png"))
