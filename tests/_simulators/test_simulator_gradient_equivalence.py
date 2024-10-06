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
    connector = pq.JaxConnector()

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
            connector=connector,
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
    connector = pq.JaxConnector()

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
            connector=connector,
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


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.SamplingSimulator))
def test_PostSelectPhotons_gradient(SimulatorClass):
    connector = pq.JaxConnector()
    state_vector = np.sqrt([0.2, 0.3, 0.5])

    def _calculate_loss(weights):
        np = connector.np

        with pq.Program() as preparation:
            pq.Q(all) | pq.StateVector([0, 1, 0]) * state_vector[0]
            pq.Q(all) | pq.StateVector([1, 1, 0]) * state_vector[1]
            pq.Q(all) | pq.StateVector([2, 1, 0]) * state_vector[2]

        phase_shifter_phis = weights[:3]
        thetas = weights[3:6]
        phis = weights[6:]
        with pq.Program() as interferometer:
            for i in range(3):
                pq.Q(i) | pq.Phaseshifter(phase_shifter_phis[i])

            pq.Q(1, 2) | pq.Beamsplitter(theta=thetas[0], phi=phis[0])
            pq.Q(0, 1) | pq.Beamsplitter(theta=thetas[1], phi=phis[1])
            pq.Q(1, 2) | pq.Beamsplitter(theta=thetas[2], phi=phis[2])

        with pq.Program() as program:
            pq.Q(all) | preparation

            pq.Q(all) | interferometer

            pq.Q(all) | pq.PostSelectPhotons(
                postselect_modes=(1, 2), photon_counts=(1, 0)
            )

        simulator = SimulatorClass(d=3, config=pq.Config(cutoff=4), connector=connector)

        state = simulator.execute(program).state

        norm = state.norm

        density_matrix = state.density_matrix[:3, :3] / norm

        expected_state = np.copy(state_vector)
        expected_state = connector.assign(expected_state, 2, -expected_state[2])

        loss = 1 - np.sqrt(
            np.real(np.conj(expected_state) @ density_matrix @ expected_state)
        )

        return loss

    weights = np.array(
        [np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, -np.pi / 8, 0, 0, 0]
    )

    loss = _calculate_loss(weights=weights)

    loss_grad = grad(_calculate_loss)

    gradient = loss_grad(weights)

    assert np.isclose(loss, 0.0)

    assert np.allclose(gradient, 0.0, atol=1e-7)


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.SamplingSimulator))
def test_ImperfectPostSelectPhotons_gradient(SimulatorClass):

    connector = pq.JaxConnector()

    state_vector = np.sqrt([0.2, 0.3, 0.5])

    def _calculate_loss(weights):
        np = connector.np

        with pq.Program() as preparation:
            pq.Q(all) | pq.StateVector([0, 1, 0]) * state_vector[0]
            pq.Q(all) | pq.StateVector([1, 1, 0]) * state_vector[1]
            pq.Q(all) | pq.StateVector([2, 1, 0]) * state_vector[2]

        phase_shifter_phis = weights[:3]
        thetas = weights[3:6]
        phis = weights[6:]
        with pq.Program() as interferometer:
            for i in range(3):
                pq.Q(i) | pq.Phaseshifter(phase_shifter_phis[i])

            pq.Q(1, 2) | pq.Beamsplitter(theta=thetas[0], phi=phis[0])
            pq.Q(0, 1) | pq.Beamsplitter(theta=thetas[1], phi=phis[1])
            pq.Q(1, 2) | pq.Beamsplitter(theta=thetas[2], phi=phis[2])

        with pq.Program() as program:
            pq.Q(all) | preparation

            pq.Q(all) | interferometer

            pq.Q(all) | pq.ImperfectPostSelectPhotons(
                postselect_modes=(1, 2),
                photon_counts=(1, 0),
                detector_efficiency_matrix=np.array(
                    [
                        [1.0, 0.2, 0.1],
                        [0.0, 0.8, 0.2],
                        [0.0, 0.0, 0.7],
                    ]
                ),
            )

        simulator = SimulatorClass(d=3, config=pq.Config(cutoff=4), connector=connector)

        state = simulator.execute(program).state

        norm = state.norm

        density_matrix = state.density_matrix[:3, :3] / norm

        expected_state = np.copy(state_vector)
        expected_state = connector.assign(expected_state, 2, -expected_state[2])

        loss = 1 - np.sqrt(
            np.real(np.conj(expected_state) @ density_matrix @ expected_state)
        )

        return loss

    weights = np.array(
        [np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, -np.pi / 8, 0, 0, 0]
    )

    loss = _calculate_loss(weights=weights)

    loss_gradient = grad(_calculate_loss)

    gradient = loss_gradient(weights)

    assert np.isclose(loss, 0.0485518312033707)

    assert np.allclose(
        gradient,
        [
            0.0,
            0.0,
            0.0,
            0.08767547,
            0.01909362,
            -0.01343052,
            0.0,
            0.0,
            0.0,
        ],
        atol=1e-7,
    )
