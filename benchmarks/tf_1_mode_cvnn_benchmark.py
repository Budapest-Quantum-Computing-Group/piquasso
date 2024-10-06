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


@pytest.fixture
def cutoff():
    return 20


@pytest.fixture
def layer_count():
    return 5


@pytest.fixture
def weights(layer_count):
    passive_sd = 0.1
    active_sd = 0.001

    sq_r = tf.random.normal(shape=[layer_count], stddev=active_sd)
    sq_phi = tf.random.normal(shape=[layer_count], stddev=passive_sd)

    d_r = tf.random.normal(shape=[layer_count], stddev=active_sd)
    d_phi = tf.random.normal(shape=[layer_count], stddev=passive_sd)

    r1 = tf.random.normal(shape=[layer_count], stddev=passive_sd)
    r2 = tf.random.normal(shape=[layer_count], stddev=passive_sd)

    kappa = tf.random.normal(shape=[layer_count], stddev=active_sd)

    weights = tf.convert_to_tensor(
        [r1, sq_r, sq_phi, r2, d_r, d_phi, kappa], dtype=np.float64
    )

    return tf.Variable(tf.transpose(weights))


def piquasso_benchmark(benchmark, weights, cutoff, layer_count):
    benchmark(lambda: _calculate_piquasso_results(weights, cutoff, layer_count))


def strawberryfields_benchmark(benchmark, weights, cutoff, layer_count):
    benchmark(lambda: _calculate_strawberryfields_results(weights, cutoff, layer_count))


def test_state_vector_and_jacobian(weights, cutoff, layer_count):
    pq_state_vector, pq_jacobian = _calculate_piquasso_results(
        weights, cutoff, layer_count
    )
    sf_state_vector, sf_jacobian = _calculate_strawberryfields_results(
        weights, cutoff, layer_count
    )

    assert np.allclose(pq_state_vector, sf_state_vector)
    assert np.allclose(pq_jacobian, sf_jacobian)


@tf.function
def _calculate_piquasso_results(weights, cutoff, layer_count):
    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=cutoff),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(0) | pq.Vacuum()
            for i in range(layer_count):
                pq.Q(0) | pq.Phaseshifter(weights[i, 0])
                pq.Q(0) | pq.Squeezing(weights[i, 1], weights[i, 2])
                pq.Q(0) | pq.Phaseshifter(weights[i, 3])
                pq.Q(0) | pq.Displacement(weights[i, 4], weights[i, 5])
                pq.Q(0) | pq.Kerr(weights[i, 6])

        state = simulator.execute(program).state
        state_vector = state.state_vector

    return state_vector, tape.jacobian(state_vector, weights)


def _calculate_strawberryfields_results(weights, cutoff, layer_count, sf):
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(1)

    with tf.GradientTape() as tape:
        with prog.context as q:
            for i in range(layer_count):
                sf.ops.Rgate(weights[i, 0]) | q[0]
                sf.ops.Sgate(weights[i, 1], weights[i, 2]) | q[0]
                sf.ops.Rgate(weights[i, 3]) | q[0]
                sf.ops.Dgate(weights[i, 4], weights[i, 5]) | q[0]
                sf.ops.Kgate(weights[i, 6]) | q[0]

        state = eng.run(prog).state
        state_vector = state.ket()

    return state_vector, tape.jacobian(state_vector, weights)
