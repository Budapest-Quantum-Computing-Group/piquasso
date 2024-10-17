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

"""
Benchmarking the Piquasso and TheWalrus torontonian implementations.
"""

import time

import piquasso as pq

import json

from piquasso._math.torontonian import torontonian as pq_torontonian
from thewalrus import rec_torontonian as tw_torontonian

import matplotlib.pyplot as plt

from scipy.stats import unitary_group

import numpy as np

np.set_printoptions(suppress=True, linewidth=200)


if __name__ == "__main__":
    x = []
    y = []
    z = []

    ITER = 100

    FILENAME = f"torontonian_benchmark_{int(time.time())}.json"

    for d in range(3, 30):
        print(d)
        x.append(d)

        simulator = pq.GaussianSimulator(d=d)

        program = pq.Program(
            instructions=[pq.Vacuum()]
            + [pq.Squeezing(r=np.random.rand()).on_modes(i) for i in range(d)]
            + [pq.Interferometer(unitary_group.rvs(d))]
        )

        state = simulator.execute(program).state

        xpxp_covariance_matrix = state.xpxp_covariance_matrix

        sigma: np.ndarray = (xpxp_covariance_matrix / 2 + np.identity(2 * d)) / 2

        input_matrix = np.identity(len(sigma), dtype=float) - np.linalg.inv(sigma)

        sum_ = 0.0

        print(pq_torontonian(input_matrix))
        for _ in range(ITER):
            print("|", end="", flush=True)
            start_time = time.time()
            pq_torontonian(input_matrix)
            sum_ += time.time() - start_time

        y.append(sum_ / ITER)
        print()

        xxpp_covariance_matrix = state.xxpp_covariance_matrix

        sigma: np.ndarray = (xxpp_covariance_matrix / 2 + np.identity(2 * d)) / 2

        input_matrix = np.identity(len(sigma), dtype=float) - np.linalg.inv(sigma)

        sum_ = 0.0

        print(tw_torontonian(input_matrix))
        for _ in range(ITER):
            print("-", end="", flush=True)
            start_time = time.time()
            tw_torontonian(input_matrix)
            sum_ += time.time() - start_time

        z.append(sum_ / ITER)
        print()

        with open(FILENAME, "w") as f:
            json.dump(dict(x=x, y=y, z=z), f, indent=4)

    plt.scatter(x, y, label="Piquasso")
    plt.scatter(x, z, label="TheWalrus")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("d [-]")
    plt.ylabel("Execution time [s]")

    plt.show()
