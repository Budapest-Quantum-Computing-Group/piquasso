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

import time

import jax

import piquasso as pq

from functools import partial

from scipy.special import comb

import numpy as np

from jax import grad, jit


@partial(jit, static_argnums=2)
def _calculate_loss(target_state_vector, weights, cutoff):
    d = pq.cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff, dtype=np.float32),
        connector=pq.JaxConnector(),
    )

    program = pq.cvqnn.create_program(weights)

    state_vector = simulator.execute(program).state.state_vector

    return jax.numpy.sum(jax.numpy.abs(state_vector - target_state_vector))


if __name__ == "__main__":
    cutoff = 10
    layer_count = 3
    d = 2

    state_vector_size = comb(d + cutoff - 1, cutoff - 1, exact=True)

    target_state_vector = np.random.rand(state_vector_size) + 1j * np.random.rand(
        state_vector_size
    )

    target_state_vector /= np.abs(target_state_vector)

    iterations = 100

    _calculate_loss_grad = jit(grad(_calculate_loss), static_argnums=2)

    weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count, d)

    start_time = time.time()
    _calculate_loss(target_state_vector, weights, cutoff)
    _calculate_loss_grad(target_state_vector, weights, cutoff)
    print("COMPILATION TIME: ", time.time() - start_time)

    runtimes = []

    for i in range(iterations):
        weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count, d)

        start_time = time.time()
        loss = _calculate_loss(target_state_vector, weights, cutoff)
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)

        print(f"{i}:\truntime = {runtime},\tloss = {loss}")

    print(f"AVERAGE RUNTIME: {np.mean(runtimes)} (+/- {np.std(runtimes)})")

    runtimes = []

    for i in range(iterations):
        weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count, d)

        start_time = time.time()
        _calculate_loss_grad(target_state_vector, weights, cutoff)
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)

        print(f"{i}:\truntime = {runtime}")

    print(f"AVERAGE RUNTIME: {np.mean(runtimes)} (+/- {np.std(runtimes)})")
