#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

from functools import partial

import jax

import numpy as np

import jax.numpy as jnp

from piquasso.fermionic._utils import fock_to_binary_indices


import optax


CONNECTOR = pq.JaxConnector()


def get_two_mode_instructions(weights, GateClass):
    instructions = []

    no_weights = len(weights)

    for i in range(no_weights):
        instructions.append(GateClass(weights[i]).on_modes(i, i + 1))

    return instructions


def get_single_mode_instructions(weights, GateClass):
    instructions = []

    no_weights = len(weights)

    for i in range(no_weights):
        instructions.append(GateClass(weights[i]).on_modes(i))

    return instructions


@partial(jax.jit, static_argnames=("d", "no_layers"))
def get_state_vector(weights, d, no_layers):
    instructions = [pq.StateVector([0] * d)]

    no_of_weights_per_layer = 3 * d - 2

    for i in range(no_layers):
        current_layer_idx = i * no_of_weights_per_layer

        xx_weights = weights[current_layer_idx : current_layer_idx + (d - 1)]
        rz_weights = weights[
            current_layer_idx + (d - 1) : current_layer_idx + 2 * d - 1
        ]
        cp_weights = weights[
            current_layer_idx + 2 * d - 1 : current_layer_idx + no_of_weights_per_layer
        ]

        xx_instructions = get_two_mode_instructions(xx_weights, pq.fermionic.IsingXX)
        rz_instructions = get_single_mode_instructions(rz_weights, pq.fermionic.RZ)
        cp_instructions = get_two_mode_instructions(
            cp_weights, pq.fermionic.ControlledPhase
        )
        instructions.extend(xx_instructions + rz_instructions + cp_instructions)

    program = pq.Program(instructions)

    simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=(d + 1)), connector=CONNECTOR
    )

    state = simulator.execute(program).state

    indices = fock_to_binary_indices(d)

    return state.state_vector[indices]


@partial(jax.jit, static_argnames=("d", "no_layers"))
def get_probabilities(weights, d, no_layers):
    state_vector = get_state_vector(weights, d, no_layers)

    return jnp.real(jnp.abs(state_vector) ** 2)


@partial(jax.jit, static_argnames=("d", "no_layers"))
def get_marginal_probabilities(weights, d, no_layers):
    """
    Here, we trace out the first qubit.
    """
    probabilities = get_probabilities(weights, d, no_layers)

    dim_half = 2 ** (d - 1)

    return probabilities[:dim_half] + probabilities[dim_half:]


@partial(jax.jit, static_argnames=("d", "no_layers"))
def get_loss(weights, target_probabilities, d, no_layers):
    probabilities = get_marginal_probabilities(weights, d, no_layers)

    return jnp.sum(jax.scipy.special.kl_div(probabilities, target_probabilities))


def get_optimized_state_vector(
    target_probabilities,
    d,
    learning_rate=0.5,
    no_max_iterations=100,
    no_layers=3,
    threshold=0.01,
):
    no_of_weights_per_layer = (d - 1) + d + (d - 1)
    no_of_weights = no_layers * no_of_weights_per_layer

    weights = 0.01 * jnp.array(np.random.rand(no_of_weights))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    get_loss_and_grad = jax.value_and_grad(get_loss)

    for _ in range(no_max_iterations):
        loss, loss_grad = get_loss_and_grad(
            weights,
            target_probabilities,
            d,
            no_layers,
        )

        updates, opt_state = optimizer.update(loss_grad, opt_state)
        weights = optax.apply_updates(weights, updates)

        if loss < threshold:
            break

    return get_state_vector(weights, d, no_layers)


if __name__ == "__main__":
    d = 5
    target_probabilities = jnp.array(np.random.rand(2 ** (d - 1)))
    target_probabilities /= np.sum(target_probabilities)

    # Warmup
    state_vector = get_optimized_state_vector(target_probabilities, d=d)

    import time

    start_time = time.time()
    state_vector = get_optimized_state_vector(target_probabilities, d=d)
    print("RUNTIME:", time.time() - start_time)
