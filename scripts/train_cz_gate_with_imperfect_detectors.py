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
import time

from piquasso.decompositions.clements import (
    get_weights_from_interferometer,
    get_interferometer_from_weights,
)
from piquasso._math.indices import get_operator_index

from typing import Callable


LR = 0.00025
ITERATIONS = 1000
ALPHA = 10
BETA = 10_000
S_STAR = 0.075

tf.get_logger().setLevel("ERROR")


def get_expected_density_matrix(
    state_vector: tf.Tensor, cutoff: int, connector: pq.TensorflowConnector
) -> tf.Tensor:
    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    config = pq.Config(normalize=False, cutoff=cutoff)
    expected_program = pq.Program(
        instructions=[
            pq.StateVector(state_00, coefficient=state_vector[0]),
            pq.StateVector(state_01, coefficient=state_vector[1]),
            pq.StateVector(state_10, coefficient=state_vector[2]),
            pq.StateVector(state_11, coefficient=-state_vector[3]),
        ]
    )

    simulator = pq.PureFockSimulator(d=4, config=config, connector=connector)
    expected_state = simulator.execute(expected_program).state

    return expected_state.density_matrix


def get_initial_weights():
    connector = pq.NumpyConnector()
    modes = (0, 2, 4, 5)

    U = np.array(
        [
            [-1 / 3, -np.sqrt(2) / 3, np.sqrt(2) / 3, 2 / 3],
            [np.sqrt(2) / 3, -1 / 3, -2 / 3, np.sqrt(2) / 3],
            [
                -np.sqrt(3 + np.sqrt(6)) / 3,
                np.sqrt(3 - np.sqrt(6)) / 3,
                -np.sqrt((3 + np.sqrt(6)) / 2) / 3,
                np.sqrt(1 / 6 - 1 / (3 * np.sqrt(6))),
            ],
            [
                -np.sqrt(3 - np.sqrt(6)) / 3,
                -np.sqrt(3 + np.sqrt(6)) / 3,
                -np.sqrt(1 / 6 - 1 / (3 * np.sqrt(6))),
                -np.sqrt((3 + np.sqrt(6)) / 2) / 3,
            ],
        ]
    )

    V = connector.embed_in_identity(U, get_operator_index(modes), 6)

    return get_weights_from_interferometer(V, connector)


def _calculate_loss(
    weights: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    P: tf.Tensor,
    connector: pq.TensorflowConnector,
    state_vector: tf.Tensor,
    cutoff: int,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    S_star: tf.Tensor,
):
    d = 6
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = connector.np

    ancilla_modes = (4, 5)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    ancilla_state = [1, 1]

    interferometer = get_interferometer_from_weights(
        weights, d, connector, config.complex_dtype
    )

    program = pq.Program(
        instructions=[
            pq.StateVector(state_00 + ancilla_state, coefficient=state_vector[0]),
            pq.StateVector(state_01 + ancilla_state, coefficient=state_vector[1]),
            pq.StateVector(state_10 + ancilla_state, coefficient=state_vector[2]),
            pq.StateVector(state_11 + ancilla_state, coefficient=state_vector[3]),
            pq.Interferometer(interferometer),
            pq.ImperfectPostSelectPhotons(
                postselect_modes=ancilla_modes,
                photon_counts=ancilla_state,
                detector_efficiency_matrix=P,
            ),
        ]
    )

    simulator = pq.PureFockSimulator(d=d, config=config, connector=connector)

    state = simulator.execute(program).state

    density_matrix = state.density_matrix
    success_prob = np.real(np.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    fidelity = np.real(np.trace(normalized_density_matrix @ expected_density_matrix))
    loss = (
        1
        - np.sqrt(fidelity)
        + alpha * np.log(1 + np.exp(-beta * (success_prob - S_star))) / beta
    )

    return loss, success_prob, fidelity


def train_step(
    weights: tf.Tensor,
    P: tf.Tensor,
    connector: pq.TensorflowConnector,
    expected_density_matrix: tf.Tensor,
    state_vector: tf.Tensor,
    cutoff: int,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    S_star: tf.Tensor,
):
    with tf.GradientTape() as tape:
        loss, success_prob, fidelity = _calculate_loss(
            weights=weights,
            P=P,
            connector=connector,
            expected_density_matrix=expected_density_matrix,
            state_vector=state_vector,
            cutoff=cutoff,
            alpha=alpha,
            beta=beta,
            S_star=S_star,
        )

    grad = tape.gradient(loss, weights)

    return loss, success_prob, fidelity, grad


def train(
    _train_step: Callable,
    weights: tf.Tensor,
    P: tf.Tensor,
    state_vector: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    connector: pq.TensorflowConnector,
    cutoff: int,
):
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    for _ in range(ITERATIONS):
        loss, success_prob, fidelity, grad = _train_step(
            weights=weights,
            P=P,
            connector=connector,
            expected_density_matrix=expected_density_matrix,
            state_vector=state_vector,
            cutoff=cutoff,
            alpha=ALPHA,
            beta=BETA,
            S_star=S_STAR,
        )

        opt.apply_gradients(zip([grad], [weights]))

    print(f"Loss: {loss}")
    print(f"success probability: {success_prob}")
    print(f"Fidelity: {fidelity}")
    print(f"Weights: {weights.numpy()}")


def main() -> None:
    cutoff = 5

    decorator = tf.function(jit_compile=True)
    connector = pq.TensorflowConnector(decorate_with=decorator)
    np = connector.np
    fallback_np = connector.fallback_np

    P = fallback_np.array(
        [
            [1.0, 0.1050, 0.0110, 0.0012, 0.001, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.8950, 0.2452, 0.0513, 0.0097, 0.0017, 0.0003, 0.0001, 0.0],
            [0.0, 0.0, 0.7438, 0.3770, 0.1304, 0.0384, 0.0104, 0.0027, 0.0007],
            [0.0, 0.0, 0.0, 0.5706, 0.4585, 0.2361, 0.0996, 0.0375, 0.0132],
            [0.0, 0.0, 0.0, 0.0, 0.4013, 0.4672, 0.3346, 0.1907, 0.0952],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2565, 0.4076, 0.3870, 0.2862],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1476, 0.3066, 0.3724],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0755, 0.1985],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9338],
        ]
    )

    P = P[:cutoff, :cutoff]
    P = tf.convert_to_tensor(P)

    state_vector = np.sqrt([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    expected_density_matrix = get_expected_density_matrix(
        state_vector=state_vector, cutoff=cutoff, connector=connector
    )
    expected_density_matrix = tf.convert_to_tensor(expected_density_matrix)

    initial_weights = get_initial_weights()

    weights = tf.Variable(initial_weights.copy(), dtype=tf.float64)

    _train_step = decorator(train_step)

    start = time.time()
    train(
        _train_step=_train_step,
        weights=weights,
        P=P,
        state_vector=state_vector,
        expected_density_matrix=expected_density_matrix,
        connector=connector,
        cutoff=cutoff,
    )
    end = time.time()
    print(f"Took {end - start} seconds")


if __name__ == "__main__":
    main()
