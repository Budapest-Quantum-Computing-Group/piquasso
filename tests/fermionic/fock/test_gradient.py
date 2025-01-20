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

import piquasso as pq
from piquasso.decompositions.clements import (
    get_weights_from_interferometer,
    get_interferometer_from_weights,
)

import numpy as np

import jax.numpy as jnp

from jax import jit, jacrev


def test_Interferometer_2_by_2_gradient():
    connector = pq.JaxConnector()
    theta = np.pi / 5

    @jit
    def calculate_state_vector(theta):
        U = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
        )

        with pq.Program() as program:
            pq.Q(0, 1) | pq.StateVector([0, 1])
            pq.Q(0, 1) | pq.Interferometer(U)

        fock_simulator = pq.fermionic.PureFockSimulator(d=2, connector=connector)

        fock_state = fock_simulator.execute(program).state

        return fock_state.state_vector

    state_vector = calculate_state_vector(np.pi / 5)

    assert np.allclose(state_vector, [0, -np.sin(theta), np.cos(theta), 0])

    calculate_state_vector_jac = jit(jacrev(calculate_state_vector, holomorphic=True))

    # NOTE: `holomorphic=True` requrires both complex inputs and outputs
    state_vector_jac = calculate_state_vector_jac(theta + 0.0j)

    assert np.allclose(state_vector_jac, [0, -np.cos(theta), -np.sin(theta), 0])


@pytest.mark.monkey
def test_Interferometer_3_by_3_random(generate_unitary_matrix):
    connector = pq.JaxConnector()

    d = 3
    U = generate_unitary_matrix(d)

    @jit
    def calculate_state_vector(U):
        with pq.Program() as program:
            # NOTE: This violates the parity superselection rule, but the simulator
            # should permit it.
            pq.Q(0, 1, 2) | pq.StateVector([0, 0, 1]) / np.sqrt(2)
            pq.Q(0, 1, 2) | pq.StateVector([0, 1, 1]) / np.sqrt(2)
            pq.Q(0, 1, 2) | pq.Interferometer(U)

        fock_simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

        fock_state = fock_simulator.execute(program).state

        return fock_state.state_vector

    state_vector = calculate_state_vector(U)

    assert np.allclose(
        state_vector,
        1
        / np.sqrt(2)
        * np.array(
            [
                0.0,
                U[0, 2],
                U[1, 2],
                U[2, 2],
                U[0, 1] * U[1, 2] - U[0, 2] * U[1, 1],
                U[0, 1] * U[2, 2] - U[0, 2] * U[2, 1],
                U[1, 1] * U[2, 2] - U[1, 2] * U[2, 1],
                0.0,
            ]
        ),
    )

    calculate_state_vector_jac = jit(jacrev(calculate_state_vector, holomorphic=True))

    state_vector_jac = calculate_state_vector_jac(U)

    def E(i, j):
        # Utility function for basis matrices.
        matrix = np.zeros(shape=(d, d))
        matrix[i, j] = 1.0
        return matrix

    assert np.allclose(state_vector_jac[0], np.zeros(shape=(d, d)))
    assert np.allclose(state_vector_jac[7], np.zeros(shape=(d, d)))

    assert np.allclose(state_vector_jac[1], E(0, 2) / np.sqrt(2))
    assert np.allclose(state_vector_jac[2], E(1, 2) / np.sqrt(2))
    assert np.allclose(state_vector_jac[3], E(2, 2) / np.sqrt(2))

    assert np.allclose(
        state_vector_jac[4],
        (E(0, 1) * U[1, 2] + U[0, 1] * E(1, 2) - E(0, 2) * U[1, 1] - U[0, 2] * E(1, 1))
        / np.sqrt(2),
    )

    assert np.allclose(
        state_vector_jac[5],
        (E(0, 1) * U[2, 2] + U[0, 1] * E(2, 2) - E(0, 2) * U[2, 1] - U[0, 2] * E(2, 1))
        / np.sqrt(2),
    )

    assert np.allclose(
        state_vector_jac[6],
        (E(1, 1) * U[2, 2] + U[1, 1] * E(2, 2) - E(1, 2) * U[2, 1] - U[1, 2] * E(2, 1))
        / np.sqrt(2),
    )


@pytest.mark.monkey
def test_parametrized_circuit_gradient_clements_random(generate_unitary_matrix):
    connector = pq.JaxConnector()

    d = 4

    @jit
    def calculate_state_vector(weights):
        U = get_interferometer_from_weights(weights.real, d, connector, np.complex128)

        with pq.Program() as program:
            pq.Q() | pq.StateVector([0, 0, 1, 1]) / np.sqrt(2)
            pq.Q() | pq.StateVector([1, 1, 0, 0]) / np.sqrt(2)
            pq.Q() | pq.Interferometer(U)

        fock_simulator = pq.fermionic.PureFockSimulator(
            d=d, connector=connector, config=pq.Config(cutoff=d + 1)
        )

        fock_state = fock_simulator.execute(program).state

        return fock_state.state_vector

    U = generate_unitary_matrix(d)

    weights = get_weights_from_interferometer(U, connector)

    state_vector = calculate_state_vector(weights)

    assert np.isclose(
        state_vector.conj() @ state_vector, 1.0
    ), "The resulting state vector should be normalized."

    assert state_vector.shape == (2**d,)

    calculate_state_vector_jac = jit(jacrev(calculate_state_vector, holomorphic=True))

    # NOTE: `holomorphic=True` requrires both complex inputs and outputs
    state_vector_jac = calculate_state_vector_jac(weights + 0.0j)

    assert state_vector_jac.shape == (2**d, len(weights))

    for i in range(2**d):
        np.isclose(
            state_vector.conj() @ state_vector_jac[:, i]
            + state_vector_jac[:, i].conj() @ state_vector,
            0.0,
        ), "Gradient of state vector norm should be 0."
