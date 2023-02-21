#
# Copyright 2021-2023 Budapest Quantum Computing Group
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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import time
import numpy as np
import json
import strawberryfields as sf
from strawberryfields import ops

displacement_input = 0.01

MIN_MODE_NUM = 2
MAX_MODE_NUM = 8
cutoff = 7


def sf_interferometer(params, q):
    N = len(q)
    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

    if N == 1:
        ops.Rgate(rphi[0]) | q[0]
        return

    n = 0

    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            if (l + k) % 2 != 1:
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

    sf_interferometer(int1, q)

    for i in range(N):
        ops.Sgate(s[i]) | q[i]

    sf_interferometer(int2, q)

    for i in range(N):
        ops.Dgate(dr[i], dp[i]) | q[i]
        ops.Kgate(k[i]) | q[i]


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat(
        [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights],
        axis=1,
    )

    weights = tf.Variable(weights)

    return weights


def strawberryfields_mean_position_benchmark(d: int):
    modes = d
    layers = 1

    target_state = np.zeros([cutoff] * modes)
    target_state[([1] + [0] * (modes - 1))] = 1.0
    target_state = tf.constant(target_state, dtype=tf.complex64)

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    qnn = sf.Program(modes)

    weights = init_weights(modes, layers)
    num_params = np.prod(weights.shape)

    sf_params = np.arange(num_params).reshape(weights.shape).astype(str)
    sf_params = np.array([qnn.params(*i) for i in sf_params])

    with qnn.context as q:
        for i in range(modes):
            sf.ops.Dgate(0.1) | q[i]

        for k in range(layers):
            layer(sf_params[k], q)

    with tf.GradientTape() as tape:
        mapping = {
            p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))
        }
        start_time = time.time()
        state = eng.run(qnn, args=mapping).state
        exec_time = time.time() - start_time
        # print("EXECUTION TIME: ", exec_time)

        ket = state.ket()

        mean, _ = state.quad_expectation(mode=0)

    start_time = time.time()
    gradient = tape.jacobian(mean, weights)
    jacobian_time = time.time() - start_time
    # print("gradient:", gradient)
    # print("JACOBIAN CALCULATION TIME: ", jacobian_time)
    return exec_time, jacobian_time


def create_layer_parameters(d: int):
    number_of_beamsplitters: int
    if d % 2 == 0:
        number_of_beamsplitters = (d // 2) ** 2
        number_of_beamsplitters += ((d - 1) // 2) * (d // 2)
    else:
        number_of_beamsplitters = ((d - 1) // 2) * d

    thetas_1 = [tf.Variable(0.1) for _ in range(number_of_beamsplitters)]
    phis_1 = [tf.Variable(0.0) for _ in range(d - 1)]

    squeezings = [tf.Variable(0.1) for _ in range(d)]

    thetas_2 = [tf.Variable(0.1) for _ in range(number_of_beamsplitters)]
    phis_2 = [tf.Variable(0.0) for _ in range(d - 1)]

    displacements = [tf.Variable(0.1) for _ in range(d)]

    kappas = [tf.Variable(0.1) for _ in range(d)]

    return {
        "d": d,
        "thetas_1": thetas_1,
        "phis_1": phis_1,
        "squeezings": squeezings,
        "thetas_2": thetas_2,
        "phis_2": phis_2,
        "displacements": displacements,
        "kappas": kappas,
    }


from piquasso._math.fock import cutoff_cardinality


def piquasso_mean_position_benchmark(d: int):
    target_state_vector = np.zeros(
        cutoff_cardinality(cutoff=cutoff, d=d), dtype=complex
    )
    target_state_vector[1] = 1.0
    target_state = tf.constant(target_state_vector, dtype=tf.complex128)

    simulator = pq.TensorflowPureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

    parameters = create_layer_parameters(d)

    with tf.GradientTape() as tape:

        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            pq.Q(all) | pq.Displacement(alpha=displacement_input)

            i = 0
            for col in range(d):
                if col % 2 == 0:
                    for mode in range(0, d - 1, 2):
                        modes = (mode, mode + 1)
                        pq.Q(*modes) | pq.Beamsplitter(
                            parameters["thetas_1"][i], phi=0.0
                        )
                        i += 1

                if col % 2 == 1:
                    for mode in range(1, d - 1, 2):
                        modes = (mode, mode + 1)
                        pq.Q(*modes) | pq.Beamsplitter(
                            parameters["thetas_1"][i], phi=0.0
                        )
                        i += 1

            for i in range(d - 1):
                pq.Q(i) | pq.Phaseshifter(parameters["phis_1"][i])

            pq.Q(all) | pq.Squeezing(parameters["squeezings"])

            i = 0
            for col in range(d):
                if col % 2 == 0:
                    for mode in range(0, d - 1, 2):
                        modes = (mode, mode + 1)
                        pq.Q(*modes) | pq.Beamsplitter(
                            parameters["thetas_2"][i], phi=0.0
                        )
                        i += 1

                if col % 2 == 1:
                    for mode in range(1, d - 1, 2):
                        modes = (mode, mode + 1)
                        pq.Q(*modes) | pq.Beamsplitter(
                            parameters["thetas_2"][i], phi=0.0
                        )
                        i += 1

            for i in range(d - 1):
                pq.Q(i) | pq.Phaseshifter(parameters["phis_2"][i])

            pq.Q(all) | pq.Displacement(alpha=parameters["displacements"])
            pq.Q(all) | pq.Kerr(parameters["kappas"])

        start_time = time.time()
        state = simulator.execute(program).state
        exec_time = time.time() - start_time
        # print("EXECUTION TIME: ", exec_time)

        mean = state.mean_position(mode=0)

    flattened_parameters = (
        parameters["thetas_1"]
        + parameters["phis_1"]
        + parameters["squeezings"]
        + parameters["thetas_2"]
        + parameters["phis_2"]
        + parameters["displacements"]
        + parameters["kappas"]
    )

    start_time = time.time()
    gradient = tape.jacobian(mean, flattened_parameters)

    jacobian_time = time.time() - start_time
    # print("JACOBIAN CALCULATION TIME:", jacobian_time)
    # print("JACOBIAN SHAPE:", [g.numpy() for g in gradient])

    return exec_time, jacobian_time


complete_dict = {"benchmarks": []}
for i in range(MIN_MODE_NUM, MAX_MODE_NUM):
    pq_exec_time, pq_jacobian_time = piquasso_mean_position_benchmark(i)
    # sf_exec_time, sf_jacobian_time = strawberryfields_mean_position_benchmark(i)
    result_dict = {
        "mode": i,
        "cutoff": cutoff,
        "pq": {"exec_time": pq_exec_time, "jacobian_time": pq_jacobian_time},
        "sf": {"exec_time": 0, "jacobian_time": 0},
    }
    out_json = open(
        "./scripts/json_dump/{}_mean-position_{}_mode.json".format(
            time.strftime("%Y%m%d-%H%M%S"), i
        ),
        "w+",
    )
    json.dump(result_dict, out_json, indent=6)
    out_json.close()
    complete_dict["benchmarks"].append(result_dict)

out_json = open(
    "./scripts/json_dump/{}_mean-position_{}-{}_modes_complete.json".format(
        time.strftime("%Y%m%d-%H%M%S"), MIN_MODE_NUM, MAX_MODE_NUM - 1
    ),
    "w+",
)
json.dump(complete_dict, out_json, indent=6)
out_json.close()
