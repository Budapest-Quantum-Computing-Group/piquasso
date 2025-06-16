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

import pytest

import numpy as np

import piquasso as pq

from jax import grad
import jax.numpy as jnp


def test_Beamsplitter_gradient_at_theta_equal_0():
    connector = pq.JaxConnector()

    def get_fidelity(theta):
        initial_state = [1, 1, 0]

        with pq.Program() as program:
            pq.Q() | pq.StateVector(initial_state)

        with pq.Program() as rotated_program:
            pq.Q() | program

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

        simulator = pq.SamplingSimulator(d=3, connector=connector)

        initial_state_vector = simulator.execute(program).state.state_vector
        rotated_state_vector = simulator.execute(rotated_program).state.state_vector

        return jnp.abs(jnp.conj(initial_state_vector) @ rotated_state_vector)

    angle = 0.0

    fidelity_at_0 = get_fidelity(angle)

    assert np.isclose(fidelity_at_0, 1.0)

    grad_get_fidelity = grad(get_fidelity)
    fidelity_grad_at_0 = grad_get_fidelity(angle)

    assert np.isclose(fidelity_grad_at_0, 0.0)


@pytest.mark.monkey
def test_Beamsplitter_gradient_at_random_angle():
    connector = pq.JaxConnector()

    def get_fidelity(theta):
        initial_state = [1, 1, 0]

        with pq.Program() as program:
            pq.Q() | pq.StateVector(initial_state)

        with pq.Program() as rotated_program:
            pq.Q() | program

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

        simulator = pq.SamplingSimulator(d=3, connector=connector)

        initial_state_vector = simulator.execute(program).state.state_vector
        rotated_state_vector = simulator.execute(rotated_program).state.state_vector

        return jnp.abs(jnp.conj(initial_state_vector) @ rotated_state_vector)

    angle = np.random.uniform() * 2 * np.pi

    expected_fidelity = np.abs(np.cos(2 * angle))

    fidelity = get_fidelity(angle)

    assert np.isclose(fidelity, expected_fidelity)

    grad_get_fidelity = grad(get_fidelity)
    fidelity_grad = grad_get_fidelity(angle)

    expected_fidelity_grad = -2 * np.sin(2 * angle) * np.sign(np.cos(2 * angle))

    assert np.isclose(fidelity_grad, expected_fidelity_grad)


@pytest.mark.monkey
def test_Beamsplitter_gradient_at_random_angle_multiparticle_initial_state():
    connector = pq.JaxConnector()

    def get_fidelity(theta):
        initial_state = [2, 1, 0]

        with pq.Program() as program:
            pq.Q() | pq.StateVector(initial_state)

        with pq.Program() as rotated_program:
            pq.Q() | program

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

        simulator = pq.SamplingSimulator(
            d=3,
            connector=connector,
            config=pq.Config(cutoff=np.sum(initial_state) + 1),
        )

        initial_state_vector = simulator.execute(program).state.state_vector
        rotated_state_vector = simulator.execute(rotated_program).state.state_vector

        return jnp.abs(jnp.conj(initial_state_vector) @ rotated_state_vector)

    angle = np.random.uniform() * 2 * np.pi

    overlap = np.cos(angle) * np.cos(2 * angle) - np.sin(angle) * np.sin(2 * angle) / 2

    expected_fidelity = np.abs(overlap)

    fidelity = get_fidelity(angle)

    assert np.isclose(fidelity, expected_fidelity)

    grad_get_fidelity = grad(get_fidelity)
    fidelity_grad = grad_get_fidelity(angle)

    overlap_grad = -2 * np.sin(angle) * np.cos(2 * angle) - 5 / 2 * np.sin(
        2 * angle
    ) * np.cos(angle)

    expected_fidelity_grad = overlap_grad * (1 if (overlap > 0) else -1)

    assert np.isclose(fidelity_grad, expected_fidelity_grad)


def test_get_particle_detection_probability_gradient():
    theta = jnp.pi / 3

    simulator = pq.SamplingSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.JaxConnector()
    )

    def func(theta):
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta)

        state = simulator.execute(program).state

        return state.get_particle_detection_probability(occupation_number=(2, 0))

    get_jacobian = grad(func)

    probability = func(theta)

    assert jnp.isclose(probability, np.cos(theta) ** 4)

    jacobian = get_jacobian(theta)

    assert jnp.isclose(jacobian, -4 * np.cos(theta) ** 3 * np.sin(theta))
