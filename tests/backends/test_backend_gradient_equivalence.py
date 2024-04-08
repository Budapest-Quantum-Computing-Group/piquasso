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

import pytest

import numpy as np

import piquasso as pq

from jax import grad
import jax.numpy as jnp


@pytest.mark.monkey
@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.SamplingSimulator))
def test_Jax_gradient_equivalence_for_single_Beamsplitter(SimulatorClass):
    calculator = pq.JaxCalculator()

    def get_fidelity(theta):
        initial_state = [1, 1, 0]

        d = len(initial_state)
        cutoff = np.sum(initial_state) + 1

        with pq.Program() as program:
            pq.Q() | pq.StateVector(initial_state)

        with pq.Program() as rotated_program:
            pq.Q() | program

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

        simulator = SimulatorClass(
            d=d,
            calculator=calculator,
            config=pq.Config(cutoff=cutoff),
        )

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


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.SamplingSimulator))
def test_Jax_gradient_equivalence_complex_scenario(SimulatorClass):
    calculator = pq.JaxCalculator()

    def get_fidelity(theta):
        initial_state = [1, 1, 0]

        d = len(initial_state)
        cutoff = np.sum(initial_state) + 1

        with pq.Program() as program:
            pq.Q() | pq.StateVector(initial_state)

        with pq.Program() as rotated_program:
            pq.Q() | program

            pq.Q(1, 2) | pq.Beamsplitter(theta=0.1, phi=0.42)

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

            pq.Q(1, 2) | pq.Beamsplitter(theta=1.5, phi=0.35)

        simulator = SimulatorClass(
            d=d,
            calculator=calculator,
            config=pq.Config(cutoff=cutoff),
        )

        initial_state_vector = simulator.execute(program).state.state_vector
        rotated_state_vector = simulator.execute(rotated_program).state.state_vector

        return jnp.abs(jnp.conj(initial_state_vector) @ rotated_state_vector)

    angle = np.pi / 7

    fidelity = get_fidelity(angle)

    expected_fidelity = 0.04604777

    assert np.isclose(fidelity, expected_fidelity)

    grad_get_fidelity = grad(get_fidelity)
    fidelity_grad = grad_get_fidelity(angle)

    expected_fidelity_grad = 0.06591828

    assert np.isclose(fidelity_grad, expected_fidelity_grad)
