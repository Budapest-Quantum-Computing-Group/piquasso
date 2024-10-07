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
import strawberryfields as sf

import time

import piquasso as pq
from piquasso import cvqnn

import tensorflow as tf

import json

tf.get_logger().setLevel("ERROR")
np.set_printoptions(suppress=True, linewidth=200)


def make_weights(layer_count, n_modes, np_seed):
    active_sd = 0.01
    passive_sd = 0.1
    np.random.seed(np_seed)

    M = int(n_modes * (n_modes - 1)) + max(1, n_modes - 1)

    int1_weights = tf.Variable(
        np.random.normal(size=(layer_count, M), loc=0.0, scale=passive_sd)
    )
    s_weights = tf.Variable(
        np.random.normal(size=(layer_count, n_modes), loc=0.0, scale=active_sd)
    )
    int2_weights = tf.Variable(
        np.random.normal(size=(layer_count, M), loc=0.0, scale=passive_sd)
    )
    dr_weights = tf.Variable(
        np.random.normal(size=(layer_count, n_modes), loc=0.0, scale=active_sd)
    )
    dp_weights = tf.Variable(
        np.random.normal(size=(layer_count, n_modes), loc=0.0, scale=passive_sd)
    )
    k_weights = tf.Variable(
        np.random.normal(size=(layer_count, n_modes), loc=0.0, scale=active_sd)
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
    n_modes = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=n_modes,
        config=pq.Config(cutoff=cutoff),
        calculator=pq.TensorflowCalculator(),
    )

    program = cvqnn.create_program(weights)

    state = simulator.execute(program).state

    return state.state_vector


def _calculate_piquasso_results(weights, n_modes, cutoff):
    with tf.GradientTape() as tape:
        state_vector = _pq_state_vector(weights, cutoff)
        loss = tf.math.reduce_mean(
            tf.math.abs(
                state_vector
                - tf.ones(len(state_vector)) / tf.math.sqrt(float(len(state_vector)))
            )
            ** 2
        )

    return state_vector, loss, tape.gradient(loss, weights)


def _compiled_pq_state_vector(weights, cutoff):
    n_modes = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=n_modes,
        config=pq.Config(cutoff=cutoff),
        calculator=pq.TensorflowCalculator(decorate_with=tf.function),
    )

    program = cvqnn.create_program(weights)

    state = simulator.execute(program).state

    return state.state_vector


@tf.function
def _compiled_calculate_piquasso_results(weights, n_modes, cutoff):
    with tf.GradientTape() as tape:
        state_vector = _compiled_pq_state_vector(weights, cutoff)
        loss = tf.math.reduce_mean(
            tf.math.abs(
                state_vector
                - tf.ones(len(state_vector)) / tf.math.sqrt(float(len(state_vector)))
            )
            ** 2
        )

    return state_vector, loss, tape.gradient(loss, weights)


def _calculate_strawberryfields_results(weights, n_modes, cutoff):
    layer_count = weights.shape[0]
    n_modes = cvqnn.get_number_of_modes(weights.shape[1])

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    qnn = sf.Program(n_modes)

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
        loss = tf.math.reduce_mean(
            (
                tf.ones(state_vector.shape)
                / tf.math.sqrt(float(tf.math.reduce_prod(state_vector.shape)))
                - tf.math.real(state_vector)
            )
            ** 2
        )

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
    timestamp = int(time.time())

    filename = f"cvqnn_comparison_data_{timestamp}.json"

    cutoff = 10
    runs = 10
    n_layers = 4
    np_seed = 42

    CALC_PQ = True
    CALC_SF = True

    data = dict(
        cutoff=cutoff,
        runs=runs,
        n_layers=n_layers,
        np_seed=np_seed,
    )

    for n_modes in range(1, 9):
        weights = make_weights(layer_count=n_layers, n_modes=n_modes, np_seed=np_seed)

        # Warmup
        if CALC_PQ:
            _ = _calculate_piquasso_results(weights, n_modes, cutoff)

        _ = _compiled_calculate_piquasso_results(weights, n_modes, cutoff)
        if CALC_SF:
            _ = _calculate_strawberryfields_results(weights, n_modes, cutoff)

        pq_runtimes = []
        pq_compiled_runtimes = []
        sf_runtimes = []

        print()

        for k in range(runs):
            print(".", end="", flush=True)
            weights = make_weights(
                layer_count=n_layers, n_modes=n_modes, np_seed=np_seed + k + 1
            )

            if CALC_PQ:
                now = time.time()
                svec, loss, grad = _calculate_piquasso_results(weights, n_modes, cutoff)
                runtime_pq = time.time() - now
                pq_runtimes.append(runtime_pq)

            now = time.time()
            svec, loss, grad = _compiled_calculate_piquasso_results(
                weights, n_modes, cutoff
            )
            runtime_pq = time.time() - now
            pq_compiled_runtimes.append(runtime_pq)

            if CALC_SF:
                now = time.time()
                svec, loss, grad = _calculate_strawberryfields_results(
                    weights, n_modes, cutoff
                )
                runtime_sf = time.time() - now
                sf_runtimes.append(runtime_sf)

        data[n_modes] = {
            "pq": pq_runtimes,
            "pqc": pq_compiled_runtimes,
            "sf": sf_runtimes,
        }

        with open(filename, "w") as f:
            json.dump(data, f)
