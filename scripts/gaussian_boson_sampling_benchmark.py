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


import time

from scipy.stats import unitary_group
import matplotlib.pyplot as plt

import strawberryfields as sf

import json

import numpy as np

import piquasso as pq


N = 100  # number of shots


def get_strawberry_samples(squeezings, unitary):
    d = len(squeezings)

    sf_program = sf.Program(d)
    sf_engine = sf.Engine(backend="gaussian")

    with sf_program.context as q:
        for i in range(d):
            sf.ops.Sgate(squeezings[i]) | q[i]

        sf.ops.Interferometer(unitary) | tuple(q[i] for i in range(d))

        sf.ops.MeasureFock() | tuple(q[i] for i in range(d))

    return sf_engine.run(sf_program, shots=N).samples


def get_piquasso_samples(squeezings, unitary):
    d = len(squeezings)

    with pq.Program() as program:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(squeezings[i])

        pq.Q() | pq.Interferometer(unitary)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(d=d, config=pq.Config(measurement_cutoff=5 + 1))

    result = simulator.execute(program, shots=N)

    return result.samples


if __name__ == "__main__":
    FILENAME = f"gaussian_boson_sampling_{int(time.time())}.json"

    np.random.seed(123)

    # Warmup
    m = 2
    squeezings = np.arcsinh(1)
    unitary = unitary_group.rvs(m)
    get_strawberry_samples(squeezings, unitary)
    get_piquasso_samples(squeezings, unitary)
    ####

    x = []
    sf_times = []
    pq_times = []

    for m in range(2, 100):
        print("m=", m)
        x.append(m)

        squeezings = np.arcsinh(1)
        unitary = unitary_group.rvs(m)

        start_time = time.time()
        samples = get_strawberry_samples(squeezings, unitary)
        runtime = time.time() - start_time
        print("SF:", runtime)
        sf_times.append(runtime)

        start_time = time.time()
        samples = get_piquasso_samples(squeezings, unitary)
        runtime = time.time() - start_time
        print("PQ:", runtime)
        pq_times.append(runtime)

        with open(FILENAME, "w") as f:
            json.dump(dict(x=x, pq_times=pq_times, sf_times=sf_times), f, indent=4)

    plt.scatter(x, sf_times, label="Strawberry Fields")
    plt.scatter(x, pq_times, label="Piquasso")

    plt.yscale("log")

    plt.legend()
    plt.show()
