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
This code has been copied from the following website with minor modifications:
https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html
"""

import time
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops


def interferometer(params, q):
    N = len(q)
    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

    if N == 1:
        # the interferometer is a single rotation
        ops.Rgate(rphi[0]) | q[0]
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for i in range(N):
        for j, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (i + j) % 2 != 1:
                ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # apply the final local phase shifts to all modes except the last one
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


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
    # Number of interferometer parameters:
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    # Create the TensorFlow variables
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


# set the random seed
tf.random.set_seed(137)
np.random.seed(137)


# define width and depth of CV quantum neural network
modes = 7
layers = 1
cutoff = 7

# defining desired state (single photon state)
target_state = np.zeros([cutoff] * modes)
target_state[([1] + [0] * (modes - 1))] = 1.0
target_state = tf.constant(target_state, dtype=tf.complex64)


# initialize engine and program
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
qnn = sf.Program(modes)

# initialize QNN weights
weights = init_weights(modes, layers)  # our TensorFlow weights
num_params = np.prod(weights.shape)  # total number of parameters in our model


# Create array of Strawberry Fields symbolic gate arguments, matching
# the size of the weights Variable.
sf_params = np.arange(num_params).reshape(weights.shape).astype(str)
sf_params = np.array([qnn.params(*i) for i in sf_params])


# Construct the symbolic Strawberry Fields program by
# looping and applying layers to the program.
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
    print("EXECUTION TIME: ", time.time() - start_time)

    ket = state.ket()

    cost = tf.reduce_sum(tf.abs(ket - target_state))


start_time = time.time()
gradient = tape.gradient(cost, weights)
print("gradient:", gradient)
print("JACOBIAN CALCULATION TIME: ", time.time() - start_time)
