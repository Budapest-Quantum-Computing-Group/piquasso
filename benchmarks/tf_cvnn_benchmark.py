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

import piquasso as pq
from piquasso import cvqnn


np.set_printoptions(suppress=True, linewidth=200)


@pytest.fixture
def cutoff():
    return 20


@pytest.fixture
def layer_count():
    return 3


@pytest.fixture
def d():
    return 2


@pytest.fixture
def weights(layer_count, d):
    active_sd = 0.01
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


def piquasso_benchmark(benchmark, weights, cutoff):
    connector = pq.TensorflowConnector(decorate_with=tf.function)
    benchmark(lambda: _calculate_piquasso_results(weights, cutoff, connector))


def strawberryfields_benchmark(benchmark, weights, cutoff, sf):
    benchmark(lambda: _calculate_strawberryfields_results(weights, cutoff, sf))


def test_state_vector_and_jacobian(weights, cutoff, sf):
    pq_state_vector, pq_jacobian = _calculate_piquasso_results(
        weights, cutoff, pq.TensorflowConnector(decorate_with=tf.function)
    )
    sf_state_vector, sf_jacobian = _calculate_strawberryfields_results(
        weights, cutoff, sf
    )

    assert np.sum(np.abs(pq_state_vector - sf_state_vector) ** 2) < 1e-10
    assert np.sum(np.abs(pq_jacobian - sf_jacobian) ** 2) < 1e-10


def _pq_state_vector(weights, cutoff, connector):
    d = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff),
        connector=connector,
    )

    program = cvqnn.create_program(weights)

    state = simulator.execute(program).state

    return state.get_tensor_representation()


@tf.function
def _calculate_piquasso_results(weights, cutoff, connector):
    with tf.GradientTape() as tape:
        state_vector = _pq_state_vector(weights, cutoff, connector)

    return state_vector, tape.jacobian(state_vector, weights)


def _calculate_strawberryfields_results(weights, cutoff, sf):
    layer_count = weights.shape[0]
    d = cvqnn.get_number_of_modes(weights.shape[1])

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    qnn = sf.Program(d)

    num_params = np.prod(weights.shape)

    sf_params = np.arange(num_params).reshape(weights.shape).astype(str)
    sf_params = np.array([qnn.params(*i) for i in sf_params])

    with qnn.context as q:
        for k in range(layer_count):
            _sf_layer(sf_params[k], q, sf)

    with tf.GradientTape() as tape:
        mapping = {
            p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))
        }

        state = eng.run(qnn, args=mapping).state
        state_vector = state.ket()

    return state_vector, tape.jacobian(state_vector, weights)


def _sf_interferometer(params, q, sf):
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


def _sf_layer(params, q, sf):
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)

    int1 = params[:M]
    s = params[M : M + N]
    int2 = params[M + N : 2 * M + N]
    dr = params[2 * M + N : 2 * M + 2 * N]
    dp = params[2 * M + 2 * N : 2 * M + 3 * N]
    k = params[2 * M + 3 * N : 2 * M + 4 * N]

    _sf_interferometer(int1, q, sf)

    for i in range(N):
        sf.ops.Sgate(s[i]) | q[i]

    _sf_interferometer(int2, q, sf)

    for i in range(N):
        sf.ops.Dgate(dr[i], dp[i]) | q[i]
        sf.ops.Kgate(k[i]) | q[i]
