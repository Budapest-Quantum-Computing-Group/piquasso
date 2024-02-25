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
Some of the code has been copyied from
`https://strawberryfields.ai/photonics/demos/run_gate_synthesis.html`.
"""

import numpy as np

import tensorflow as tf
import strawberryfields as sf

import piquasso as pq
from piquasso import cvqnn

import time


tf.get_logger().setLevel("ERROR")
np.set_printoptions(suppress=True, linewidth=200)


def measure_graph_size(f, *args):
    if not hasattr(f, "get_concrete_function"):
        return 0

    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def _pq_loss(weights, cutoff, calculator):
    d = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff, normalize=False),
        calculator=calculator,
    )

    program = cvqnn.create_program(weights)

    state_vector = simulator.execute(program).state.state_vector

    return tf.math.reduce_mean(
        (
            tf.ones_like(state_vector) / np.sqrt(len(state_vector))
            - tf.math.real(state_vector)
        )
        ** 2
    )


@tf.function(jit_compile=True)
def _calculate_piquasso_results(weights, cutoff, calculator):
    with tf.GradientTape() as tape:
        loss = _pq_loss(weights, cutoff, calculator)

    return loss, tape.gradient(loss, weights)


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
        loss = tf.math.reduce_mean(
            (
                tf.ones_like(state_vector) / np.sqrt(len(state_vector))
                - tf.math.real(state_vector)
            )
            ** 2
        )

    return loss, tape.gradient(loss, weights)


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
    d = 2
    layer_count = 5
    cutoff = 10

    NUMBER_OF_ITERATIONS = 10

    weights = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
    )

    calculator = pq.TensorflowCalculator(decorate_with=tf.function(jit_compile=True))

    start_time = time.time()
    _calculate_piquasso_results(
        weights,
        cutoff,
        calculator,
    )
    print("PQ COMPILE TIME:", time.time() - start_time)
    print(
        "GRAPH SIZE:",
        measure_graph_size(
            _calculate_piquasso_results,
            weights,
            cutoff,
            calculator,
        ),
    )

    for i in range(NUMBER_OF_ITERATIONS):
        weights = tf.Variable(
            pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
        )
        start_time = time.time()
        print(f"{i:} ", end="")
        _calculate_piquasso_results(weights, cutoff, calculator)
        print("PQ RUNTIME:", time.time() - start_time)

    weights = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
    )

    start_time = time.time()
    _calculate_strawberryfields_results(weights, cutoff)
    print("SF COMPILE TIME:", time.time() - start_time)

    for i in range(10):
        weights = tf.Variable(
            pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count, d=d)
        )
        print(f"{i:} ", end="")
        start_time = time.time()
        _calculate_strawberryfields_results(weights, cutoff)
        print("SF RUNTIME:", time.time() - start_time)
