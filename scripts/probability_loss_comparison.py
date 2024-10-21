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
SF_CUTOFF_RANGE = range(3, 6 + 1)

ACTIVE_VAR = 0.1
PASSIVE_VAR = 1.0

WARMUP = 10
ITERATIONS = 100
d = 10
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


if __name__ == "__main__":
    print(f"ITERATIONS={ITERATIONS}, d={d}, number_of_layers={number_of_layers}")

    pq_benchmark()
    sf_benchmark()
