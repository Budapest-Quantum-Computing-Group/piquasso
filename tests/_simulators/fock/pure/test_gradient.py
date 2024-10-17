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

import jax
import jax.numpy as jnp

import piquasso as pq


def test_Beamsplitter_fock_probabilities_gradient_2_particles():
    theta = jnp.pi / 3

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.JaxConnector()
    )

    def func(theta):
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta)

        state = simulator.execute(program).state

        return state.fock_probabilities

    get_jacobian = jax.jacrev(func)

    fock_probabilities = func(theta)

    assert jnp.allclose(
        fock_probabilities,
        jnp.array(
            [
                0,
                0,
                0,
                jnp.cos(theta) ** 4,
                2 * (jnp.cos(theta) * jnp.sin(theta)) ** 2,
                jnp.sin(theta) ** 4,
            ]
        ),
    )

    jacobian = get_jacobian(theta)

    assert jnp.allclose(
        jacobian,
        jnp.array(
            [
                0,
                0,
                0,
                -4 * jnp.cos(theta) ** 3 * jnp.sin(theta),
                2 * (jnp.sin(2 * theta)) * jnp.cos(2 * theta),
                4 * jnp.sin(theta) ** 3 * jnp.cos(theta),
            ]
        ),
    )


def test_Beamsplitter_fock_probabilities_gradient_2_particles_JIT():
    theta = jnp.pi / 3

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.JaxConnector()
    )

    @jax.jit
    def func(theta):
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta)

        state = simulator.execute(program).state

        return state.fock_probabilities

    get_jacobian = jax.jacrev(func)

    fock_probabilities = func(theta)

    assert jnp.allclose(
        fock_probabilities,
        jnp.array(
            [
                0,
                0,
                0,
                jnp.cos(theta) ** 4,
                2 * (jnp.cos(theta) * jnp.sin(theta)) ** 2,
                jnp.sin(theta) ** 4,
            ]
        ),
    )

    jacobian = get_jacobian(theta)

    assert jnp.allclose(
        jacobian,
        jnp.array(
            [
                0,
                0,
                0,
                -4 * jnp.cos(theta) ** 3 * jnp.sin(theta),
                2 * (jnp.sin(2 * theta)) * jnp.cos(2 * theta),
                4 * jnp.sin(theta) ** 3 * jnp.cos(theta),
            ]
        ),
    )
