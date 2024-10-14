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
import tensorflow as tf

import strawberryfields as sf
import piquasso as pq


def pq_cvnn_gradient(weights, d, cutoff, layer_count):
    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff, normalize=True),
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
        ket = state.state_vector

    return state, ket, tape.gradient(ket, weights)


def sf_cvnn_gradient(weights, d, cutoff, layer_count):
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(d)

    with tf.GradientTape() as tape:
        with prog.context as q:
            for i in range(layer_count):
                sf.ops.Rgate(weights[i, 0]) | q[0]
                sf.ops.Sgate(weights[i, 1], weights[i, 2]) | q[0]
                sf.ops.Rgate(weights[i, 3]) | q[0]
                sf.ops.Dgate(weights[i, 4], weights[i, 5]) | q[0]
                sf.ops.Kgate(weights[i, 6]) | q[0]

        state = eng.run(prog).state
        ket = state.ket()

    return state, ket, tape.gradient(ket, weights)


def cvnn_comparison():
    cutoff = 12
    d = 1
    layer_count = 3
    tf.random.set_seed(42)
    np.random.seed(42)

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
    weights = tf.Variable(tf.transpose(weights))
    pq_state, pq_ket, pq_grad = pq_cvnn_gradient(weights, d, cutoff, layer_count)
    sf_state, sf_ket, sf_grad = sf_cvnn_gradient(weights, d, cutoff, layer_count)
    assert np.allclose(pq_ket, sf_ket)
    assert np.allclose(np.asarray(pq_grad), np.asarray(sf_grad))


if __name__ == "__main__":
    cvnn_comparison()
