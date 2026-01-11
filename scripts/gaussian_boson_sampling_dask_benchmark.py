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


import time

from scipy.stats import unitary_group
import matplotlib.pyplot as plt

import json

import numpy as np

import piquasso as pq


SHOTS = 100  # number of shots


def get_piquasso_samples(squeezings, unitary, use_dask):
    d = len(squeezings)

    with pq.Program() as program:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(squeezings[i])

        pq.Q() | pq.Interferometer(unitary)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=d, config=pq.Config(measurement_cutoff=5 + 1, use_dask=use_dask)
    )

    return simulator.execute(program, shots=SHOTS).samples


if __name__ == "__main__":
    FILENAME = f"gaussian_boson_sampling_{int(time.time())}.json"

    np.random.seed(123)

    # Warmup
    m = 2
    squeezings = np.arcsinh(np.ones(m))
    unitary = unitary_group.rvs(m)
    get_piquasso_samples(squeezings, unitary, use_dask=True)
    get_piquasso_samples(squeezings, unitary, use_dask=False)
    ####

    x = []
    pq_dask_times = []
    pq_times = []

    for m in range(25, 26):
        print("m=", m)
        x.append(m)

        squeezings = np.arcsinh(np.ones(m))
        unitary = unitary_group.rvs(m)

        start_time = time.time()
        samples = get_piquasso_samples(squeezings, unitary, use_dask=True)
        runtime = time.time() - start_time
        print("PQ (dask):", runtime)
        pq_dask_times.append(runtime)

        start_time = time.time()
        samples = get_piquasso_samples(squeezings, unitary, use_dask=False)
        runtime = time.time() - start_time
        print("PQ:", runtime)
        pq_times.append(runtime)

        with open(FILENAME, "w") as f:
            json.dump(
                dict(x=x, pq_times=pq_times, pq_dask_times=pq_dask_times), f, indent=4
            )

    plt.scatter(x, pq_dask_times, label="Piquasso (dask)")
    plt.scatter(x, pq_times, label="Piquasso")

    plt.yscale("log")

    plt.legend()
    plt.show()
