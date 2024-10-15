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

import os

import strawberryfields as sf
from strawberryfields import ops
import piquasso as pq
import time
import numpy as np
import json

from piquasso import cvqnn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

PQ_FILE_NAME = f"pq_results_{TIMESTAMP}.json"
SF_FILE_NAME = f"sf_results_{TIMESTAMP}.json"

PQ_CUTOFF_RANGE = range(3, 16 + 1)
SF_CUTOFF_RANGE = range(3, 8 + 1)

ACTIVE_VAR = 0.1
PASSIVE_VAR = 1.0

WARMUP = 10
ITERATIONS = 100
d = 8
number_of_layers = 4


def pq_benchmark():
    print("PIQUASSO_START")

    benchmark_json_list = {
        "d": d,
        "number_of_layers": number_of_layers,
        "benchmarks": [],
    }

    for cutoff in PQ_CUTOFF_RANGE:
        print(f"cutoff={cutoff}")

        simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        exec_time_list = []
        probability_lost_list = []

        for _ in range(ITERATIONS + WARMUP):
            weights = cvqnn.generate_random_cvqnn_weights(
                number_of_layers, d, active_var=ACTIVE_VAR, passive_var=PASSIVE_VAR
            )
            program = cvqnn.create_program(weights)

            start_time = time.time()
            state = simulator.execute(program).state
            exec_time = time.time() - start_time

            exec_time_list.append(exec_time)

            probability_lost_list.append(1.0 - state.norm)

        average_exec_time = np.average(exec_time_list[WARMUP:])
        average_probability_lost = np.average(probability_lost_list[WARMUP:])

        benchmark_values = {
            "cutoff": cutoff,
            "probability_lost": average_probability_lost,
            "exec_time": average_exec_time,
            "iterations": ITERATIONS,
        }

        benchmark_json_list["benchmarks"].append(benchmark_values)

        with open(PQ_FILE_NAME, "w") as file:
            json.dump(benchmark_json_list, file, indent=6)


def sf_benchmark():
    print("STRAWBERRYFIELDS_START")
    """
    This code has been copied from the following website with minor modifications:
    https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html
    """

    def interferometer(params, q):
        N = len(q)
        theta = params[: N * (N - 1) // 2]
        phi = params[N * (N - 1) // 2 : N * (N - 1)]
        rphi = params[-N + 1 :]

        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return

        n = 0

        for i in range(N):
            for j, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (i + j) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]

    def layer(params, q):
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        int1 = params[:M]
        s = params[M : M + N]
        int2 = params[M + N : 2 * M + N]
        dr = params[2 * M + N : 2 * M + 2 * N]
        dp = params[2 * M + 2 * N : 2 * M + 3 * N]
        k = params[2 * M + 3 * N : 2 * M + 4 * N]

        # begin layer
        interferometer(int1, q)

        for i in range(N):
            ops.Sgate(s[i]) | q[i]

        interferometer(int2, q)

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]

    benchmark_json_list = {
        "d": d,
        "number_of_layers": number_of_layers,
        "benchmarks": [],
    }

    for cutoff in SF_CUTOFF_RANGE:
        print(f"cutoff={cutoff}")

        exec_time_list = []
        probability_lost_list = []

        for _ in range(ITERATIONS + WARMUP):
            eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
            qnn = sf.Program(d)
            weights = cvqnn.generate_random_cvqnn_weights(
                number_of_layers, d, active_var=ACTIVE_VAR, passive_var=PASSIVE_VAR
            )

            with qnn.context as q:
                for k in range(number_of_layers):
                    layer(weights[k], q)

            start_time = time.time()
            state = eng.run(qnn).state
            exec_time = time.time() - start_time
            exec_time_list.append(exec_time)

            ket = state.ket()
            flatten_ket = np.reshape(ket, (-1))
            probability_lost = 1.0 - np.sqrt(
                np.real(np.dot(flatten_ket, np.conj(flatten_ket)))
            )

            probability_lost_list.append(probability_lost)

        average_exec_time = np.average(exec_time_list[WARMUP:])
        average_probability_lost = np.average(probability_lost_list[WARMUP:])

        benchmark_values = {
            "cutoff": cutoff,
            "probability_lost": average_probability_lost,
            "exec_time": average_exec_time,
            "iterations": ITERATIONS,
        }

        benchmark_json_list["benchmarks"].append(benchmark_values)

        with open(SF_FILE_NAME, "w") as file:
            json.dump(benchmark_json_list, file, indent=6)


def plot():
    import matplotlib.pyplot as plt

    # `pq_data` and `sf_data` are copied from result json
    pq_data = [
        {
            "cutoff": 3,
            "probability_lost": 0.06761343361309514,
            "exec_time": 0.07707272529602051,
            "iterations": 100,
        },
        {
            "cutoff": 4,
            "probability_lost": 0.026627229741442007,
            "exec_time": 0.09586937189102172,
            "iterations": 100,
        },
        {
            "cutoff": 5,
            "probability_lost": 0.009310363230509986,
            "exec_time": 0.09723606586456299,
            "iterations": 100,
        },
        {
            "cutoff": 6,
            "probability_lost": 0.0038885477163887005,
            "exec_time": 0.17679012537002564,
            "iterations": 100,
        },
        {
            "cutoff": 7,
            "probability_lost": 0.0014291706525091175,
            "exec_time": 0.27309939384460447,
            "iterations": 100,
        },
        {
            "cutoff": 8,
            "probability_lost": 0.0005489748994153664,
            "exec_time": 0.4538575768470764,
            "iterations": 100,
        },
        {
            "cutoff": 9,
            "probability_lost": 0.00018150535331517447,
            "exec_time": 0.7809635353088379,
            "iterations": 100,
        },
        {
            "cutoff": 10,
            "probability_lost": 8.020602503065799e-05,
            "exec_time": 1.3488433122634889,
            "iterations": 100,
        },
        {
            "cutoff": 11,
            "probability_lost": 2.787122983025636e-05,
            "exec_time": 2.3174398803710936,
            "iterations": 100,
        },
        {
            "cutoff": 12,
            "probability_lost": 8.19969988380298e-06,
            "exec_time": 3.89780232667923,
            "iterations": 100,
        },
        {
            "cutoff": 13,
            "probability_lost": 4.432623301681149e-06,
            "exec_time": 6.392375166416168,
            "iterations": 100,
        },
        {
            "cutoff": 14,
            "probability_lost": 2.4858229481905526e-06,
            "exec_time": 10.324620230197906,
            "iterations": 100,
        },
        {
            "cutoff": 15,
            "probability_lost": 1.7456523808623282e-06,
            "exec_time": 16.413836696147918,
            "iterations": 100,
        },
        {
            "cutoff": 16,
            "probability_lost": 3.727249816831701e-07,
            "exec_time": 25.990539412498475,
            "iterations": 100,
        },
    ]

    sf_data = [
        {
            "cutoff": 3,
            "probability_lost": 0.01899152229364203,
            "exec_time": 0.6591548776626587,
            "iterations": 100,
        },
        {
            "cutoff": 4,
            "probability_lost": 0.006241793694938461,
            "exec_time": 5.193639545440674,
            "iterations": 100,
        },
        {
            "cutoff": 5,
            "probability_lost": 0.0011602762417102263,
            "exec_time": 26.8263884472847,
            "iterations": 100,
        },
        {
            "cutoff": 6,
            "probability_lost": 0.000126302225061381,
            "exec_time": 115.39776229858398,
            "iterations": 1,
        },
        {
            "cutoff": 7,
            "probability_lost": 9.735536148403057e-05,
            "exec_time": 406.9243643283844,
            "iterations": 1,
        },
        {
            "cutoff": 8,
            "probability_lost": 1.9633707137645118e-05,
            "exec_time": 1138.993706703186,
            "iterations": 1,
        },
    ]

    pq_losses = []
    pq_exec_times = []

    for point in pq_data:
        pq_losses.append(point["probability_lost"])
        pq_exec_times.append(point["exec_time"])

    sf_losses = []
    sf_exec_times = []

    for point in sf_data:
        sf_losses.append(point["probability_lost"])
        sf_exec_times.append(point["exec_time"])

    plt.scatter(pq_exec_times, pq_losses, c="b", marker="x", label="Piquasso")
    plt.scatter(sf_exec_times, sf_losses, c="r", marker="s", label="Strawberry Fields")

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("Probability loss [-]")
    plt.xlabel("Execution time [s]")

    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    print(f"ITERATIONS={ITERATIONS}, d={d}, number_of_layers={number_of_layers}")

    pq_benchmark()
    sf_benchmark()

    plot()
