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

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import unitary_group

import piquasso as pq


SHOTS = 1000
SEED = 123

# Benchmark mode range.
MIN_D = 30
MAX_D = 40


def get_piquasso_samples(input_state: np.ndarray, unitary: np.ndarray, use_dask: bool):
    d = len(input_state)

    with pq.Program() as program:
        pq.Q() | pq.NumberState(input_state)
        pq.Q() | pq.Interferometer(unitary)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.PassiveSimulator(
        d=d,
        config=pq.Config(
            seed_sequence=SEED,
            use_dask=use_dask,
        ),
    )

    return simulator.execute(program, shots=SHOTS).samples


if __name__ == "__main__":
    filename = f"boson_sampling_{int(time.time())}.json"

    x = []
    pq_dask_times = []
    pq_times = []

    for d in range(MIN_D, MAX_D + 1, 2):
        print("d =", d)
        x.append(d)

        input_state = np.zeros(d, dtype=int)
        input_state[: d // 2] = 1
        unitary = unitary_group.rvs(d, random_state=SEED + d)
        start_time = time.time()
        get_piquasso_samples(input_state, unitary, use_dask=True)
        runtime = time.time() - start_time
        print("Piquasso PassiveSimulator with dask:", runtime)
        pq_dask_times.append(runtime)

        start_time = time.time()
        get_piquasso_samples(input_state, unitary, use_dask=False)
        runtime = time.time() - start_time
        print("Piquasso PassiveSimulator:", runtime)
        pq_times.append(runtime)

        with open(filename, "w") as f:
            json.dump(dict(pq_times=pq_times, pq_dask_times=pq_dask_times), f, indent=4)

    plt.scatter(x, pq_dask_times, label="Piquasso PassiveSimulator with dask")
    plt.scatter(x, pq_times, label="Piquasso PassiveSimulator")

    plt.xlabel("Number of modes")
    plt.ylabel("Runtime [s]")
    plt.yscale("log")

    plt.legend()
    plt.show()
