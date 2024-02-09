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

import pytest

import numpy as np
import tensorflow as tf

import strawberryfields as sf
import piquasso as pq


@pytest.fixture
def cutoff():
    return 10


@pytest.fixture
def layer_count():
    return 3


@pytest.fixture
def d():
    return 2


@pytest.fixture
def weights(layer_count, d):
    active_sd = 0.0001
    passive_sd = 0.1

    M = int(d * (d - 1)) + max(1, d - 1)

    int1_weights = tf.random.normal(shape=[layer_count, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layer_count, d], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layer_count, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layer_count, d], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layer_count, d], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layer_count, d], stddev=active_sd)

    weights = tf.cast(
        tf.concat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights],
            axis=1,
        ),
        dtype=tf.float64,
    )

    weights = tf.Variable(weights)

    return weights


def piquasso_benchmark(benchmark, weights, d, cutoff, layer_count):
    benchmark(lambda: _calculate_piquasso_results(weights, d, cutoff, layer_count))


def strawberryfields_benchmark(benchmark, weights, d, cutoff, layer_count):
    benchmark(
        lambda: _calculate_strawberryfields_results(weights, d, cutoff, layer_count)
    )


def test_state_vector_and_jacobian(weights, d, cutoff, layer_count):
    pq_state_vector, pq_jacobian = _calculate_piquasso_results(
        weights, d, cutoff, layer_count
    )
    sf_state_vector, sf_jacobian = _calculate_strawberryfields_results(
        weights, d, cutoff, layer_count
    )

    assert np.allclose(pq_state_vector, sf_state_vector)
    assert np.allclose(pq_jacobian, sf_jacobian)


def _calculate_piquasso_results(weights, d, cutoff, layer_count):
    simulator = pq.TensorflowPureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff, dtype=weights.dtype, normalize=False),
    )

    with tf.GradientTape() as tape:
        instructions = []

        for i in range(layer_count):
            instructions.extend(_pq_layer(weights[i], d))

        program = pq.Program(instructions=[pq.Vacuum()] + instructions)

        state = simulator.execute(program).state
        state_vector = state.get_tensor_representation()

    return state_vector, tape.gradient(state_vector, weights)


def _pq_interferometer(params, d):
    instructions = []

    theta = params[: d * (d - 1) // 2]
    phi = params[d * (d - 1) // 2 : d * (d - 1)]
    rphi = params[-d + 1 :]

    if d == 1:
        return [pq.Phaseshifter(rphi[0]).on_modes(0)]

    n = 0

    q = tuple(range(d))

    for j in range(d):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            if (j + k) % 2 != 1:
                instructions.append(
                    pq.Beamsplitter(theta=theta[n], phi=phi[n]).on_modes(q1, q2)
                )
                n += 1

    instructions.extend(
        [pq.Phaseshifter(rphi[i]).on_modes(i) for i in range(max(1, d - 1))]
    )

    return instructions


def _pq_layer(params, d):
    M = int(d * (d - 1)) + max(1, d - 1)

    int1 = params[:M]
    s = params[M : M + d]
    int2 = params[M + d : 2 * M + d]
    dr = params[2 * M + d : 2 * M + 2 * d]
    dp = params[2 * M + 2 * d : 2 * M + 3 * d]
    k = params[2 * M + 3 * d : 2 * M + 4 * d]

    instructions = []

    instructions.extend(_pq_interferometer(int1, d))

    instructions.extend([pq.Squeezing(s[i]).on_modes(i) for i in range(d)])

    instructions.extend(_pq_interferometer(int2, d))

    instructions.extend([pq.Displacement(dr[i], dp[i]).on_modes(i) for i in range(d)])

    instructions.extend([pq.Kerr(k[i]).on_modes(i) for i in range(d)])

    return instructions


def _calculate_strawberryfields_results(weights, d, cutoff, layer_count):
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    qnn = sf.Program(d)

    num_params = np.prod(weights.shape)

    sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
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

    return state_vector, tape.gradient(state_vector, weights)


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
