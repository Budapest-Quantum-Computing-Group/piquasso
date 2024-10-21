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

import time
import json

import strawberryfields as sf

import tensorflow as tf

import piquasso as pq

from piquasso import cvqnn

tf.get_logger().setLevel("ERROR")
np.set_printoptions(suppress=True, linewidth=200)

RUN = "pq"
GRAPH_COMPILE = True

n_layers = 4
n_mode_range = list(range(2, 15 + 1))
cutoff = 10
np_seed = 42

timestamp = int(time.time())
filename = (
    f"./result_{RUN}_graph={GRAPH_COMPILE}_"
    f"d={min(n_mode_range)}-{max(n_mode_range)}_l={n_layers}_c={cutoff}_{timestamp}"
    ".json"
)

decorator = tf.function if GRAPH_COMPILE else lambda x: x


def main():
    # Set TensorFlow to use only the CPU
    tf.config.set_visible_devices([], "GPU")

    # Verify devices
    # This step is optional, just to confirm that GPU is not being used.
    physical_devices = tf.config.list_physical_devices()
    print("Available devices:", physical_devices)

    # Create a process to run the benchmark function with queue, pq, and sf as arguments
    result = {"RUN": RUN, "cutoff": cutoff, "np_seed": np_seed, "runtimes": {}}
    with tf.device("/CPU:0"):
        for d in n_mode_range:
            partial_result = benchmark(
                n_layers=n_layers,
                d=d,
                cutoff=cutoff,
                np_seed=np_seed,
            )
            result["runtimes"][d] = partial_result

            with open(filename, "w") as json_file:
                json.dump(result, json_file, indent=4)


def benchmark(n_layers, d, cutoff, np_seed):
    weights = make_weights(layer_count=n_layers, d=d, np_seed=np_seed)

    runtimes = []

    if RUN == "pq":
        func_to_benchmark = _calculate_piquasso_results
    else:
        func_to_benchmark = _calculate_strawberryfields_results

    # Warmup
    start_time = time.time()
    func_to_benchmark(weights, cutoff)
    print("Warmup time:", time.time() - start_time)

    runs = 100
    k = 0
    while k < runs:
        print(".", end="", flush=True)
        weights = make_weights(layer_count=n_layers, d=d, np_seed=42 + k + 1)

        now = time.time()
        _, _, grad = func_to_benchmark(weights, cutoff)
        runtime = time.time() - now
        runtimes.append(runtime)

        assert grad is not None, "'grad is None', graph might be disconnected"

        if runtime > 1.0:
            runs = 10

        k = k + 1

    print()

    return runtimes


def make_weights(layer_count, d, np_seed):
    active_sd = 0.1
    passive_sd = 1.0
    np.random.seed(np_seed)

    M = int(d * (d - 1)) + max(1, d - 1)

    int1_weights = tf.Variable(
        np.random.normal(size=(layer_count, M), loc=0.0, scale=passive_sd)
    )
    s_weights = tf.Variable(
        np.random.normal(size=(layer_count, d), loc=0.0, scale=active_sd)
    )
    int2_weights = tf.Variable(
        np.random.normal(size=(layer_count, M), loc=0.0, scale=passive_sd)
    )
    dr_weights = tf.Variable(
        np.random.normal(size=(layer_count, d), loc=0.0, scale=active_sd)
    )
    dp_weights = tf.Variable(
        np.random.normal(size=(layer_count, d), loc=0.0, scale=passive_sd)
    )
    k_weights = tf.Variable(
        np.random.normal(size=(layer_count, d), loc=0.0, scale=active_sd)
    )

    weights = tf.cast(
        tf.concat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights],
            axis=1,
        ),
        dtype=tf.float64,
    )

    weights = tf.Variable(weights)

    return weights


def _pq_state_vector(weights, cutoff):
    d = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff),
        connector=pq.TensorflowConnector(
            decorate_with=(tf.function() if GRAPH_COMPILE else None)
        ),
    )

    program = cvqnn.create_program(weights)

    state = simulator.execute(program).state

    return state.state_vector


@decorator
def _calculate_piquasso_results(weights, cutoff):
    with tf.GradientTape() as tape:
        state_vector = _pq_state_vector(weights, cutoff)

        target_state_vector = tf.ones(
            shape=state_vector.shape, dtype=state_vector.dtype
        ) / tf.math.sqrt(tf.cast(len(state_vector), state_vector.dtype))

        loss = tf.math.reduce_mean(tf.abs(target_state_vector - state_vector) ** 2)

    return state_vector, loss, tape.gradient(loss, weights)


def _calculate_strawberryfields_results(weights, cutoff):
    layer_count = weights.shape[0]
    d = cvqnn.get_number_of_modes(weights.shape[1])

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    qnn = sf.Program(d)

    num_params = np.prod(weights.shape)

    sf_params = np.arange(num_params).reshape(weights.shape).astype(str)
    sf_params = np.array([qnn.params(*i) for i in sf_params])

    with qnn.context as q:
        for k in range(layer_count):
            _sf_layer(sf_params[k], q)

    with tf.GradientTape() as tape:
        mapping = {
            p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))
        }

        state = eng.run(qnn, args=mapping).state
        state_vector = state.ket()
        target_state_vector = tf.ones(
            shape=state_vector.shape, dtype=state_vector.dtype
        ) / tf.math.sqrt(tf.cast(len(state_vector), state_vector.dtype))

        loss = tf.math.reduce_mean(tf.abs(target_state_vector - state_vector) ** 2)

    return state_vector, loss, tape.gradient(loss, weights)


def _sf_interferometer(params, q):
    N = len(q)
    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

    if N == 1:
        sf.ops.Rgate(rphi[0]) | q[0]
        return

    n = 0

    for j in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            if (j + k) % 2 != 1:
                sf.ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    for i in range(max(1, N - 1)):
        sf.ops.Rgate(rphi[i]) | q[i]


def _sf_layer(params, q):
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)

    int1 = params[:M]
    s = params[M : M + N]
    int2 = params[M + N : 2 * M + N]
    dr = params[2 * M + N : 2 * M + 2 * N]
    dp = params[2 * M + 2 * N : 2 * M + 3 * N]
    k = params[2 * M + 3 * N : 2 * M + 4 * N]

    _sf_interferometer(int1, q)

    for i in range(N):
        sf.ops.Sgate(s[i]) | q[i]

    _sf_interferometer(int2, q)

    for i in range(N):
        sf.ops.Dgate(dr[i], dp[i]) | q[i]
        sf.ops.Kgate(k[i]) | q[i]


if __name__ == "__main__":
    main()
