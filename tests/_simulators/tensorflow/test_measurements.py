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

import piquasso as pq
import tensorflow as tf
import numpy as np


def test_PostSelectPhotons_gradient():
    def _calculate_loss(weights, connector, state_vector):
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

        simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=4), connector=connector
        )

        state = simulator.execute(program).state

        state.normalize()

        density_matrix = state.density_matrix[:3, :3]

        expected_state = np.copy(state_vector)
        expected_state = connector.assign(expected_state, 2, -expected_state[2])

        loss = 1 - np.sqrt(
            np.real(np.conj(expected_state) @ density_matrix @ expected_state)
        )

        return loss

    connector = pq.TensorflowConnector()

    weights = tf.Variable(
        [np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, -np.pi / 8, 0, 0, 0]
    )

    with tf.GradientTape() as tape:
        loss = _calculate_loss(
            weights=weights,
            connector=connector,
            state_vector=np.sqrt([0.2, 0.3, 0.5]),
        )

    grad = tape.gradient(loss, weights)

    assert np.isclose(loss, 0.0)

    assert np.allclose(grad, 0.0, atol=1e-7)


def test_ImperfectPostSelectPhotons_gradient():
    def _calculate_loss(weights, connector, state_vector):
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

        simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=4), connector=connector
        )

        state = simulator.execute(program).state

        state.normalize()

        density_matrix = state.density_matrix[:3, :3]

        expected_state = np.copy(state_vector)
        expected_state = connector.assign(expected_state, 2, -expected_state[2])

        loss = 1 - np.sqrt(
            np.real(np.conj(expected_state) @ density_matrix @ expected_state)
        )

        return loss

    connector = pq.TensorflowConnector()

    weights = tf.Variable(
        [np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, -np.pi / 8, 0, 0, 0]
    )

    with tf.GradientTape() as tape:
        loss = _calculate_loss(
            weights=weights,
            connector=connector,
            state_vector=np.sqrt([0.2, 0.3, 0.5]),
        )

    grad = tape.gradient(loss, weights)

    assert np.isclose(loss, 0.0485518312033707)

    assert np.allclose(
        grad,
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
