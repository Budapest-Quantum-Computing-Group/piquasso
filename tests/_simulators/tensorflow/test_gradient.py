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

import tensorflow as tf

import numpy as np

import piquasso as pq

from scipy.stats import unitary_group
from scipy.linalg import block_diag


def test_Displacement_mean_photon_number_gradient_1_mode():
    r = tf.Variable(0.43)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=6), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

        state = simulator.execute(program).state

        mean = state.mean_photon_number()

    gradient = tape.gradient(mean, [r])

    assert np.isclose(mean, r**2)
    assert np.isclose(gradient, 2 * r)


def test_Displacement_mean_photon_number_gradient_1_mode_with_phaseshift():
    r = 0.1
    phi = tf.Variable(np.pi / 5)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=8), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r, phi=phi)

        state = simulator.execute(program).state

        mean, _ = state.quadratures_mean_variance(modes=(0,))

    gradient = tape.gradient(mean, [phi])

    assert np.isclose(mean, np.sqrt(2 * simulator.config.hbar) * r * np.cos(phi))
    assert np.isclose(gradient, -np.sqrt(2 * simulator.config.hbar) * r * np.sin(phi))


def test_Displacement_mean_photon_number_gradient_2_modes():
    r = tf.Variable([0.15, 0.2])

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=6),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r[0])
            pq.Q(1) | pq.Displacement(r=r[1])

        result = simulator.execute(program)

        state = result.state

        mean = state.mean_photon_number()

    gradient = tape.gradient(mean, [r])

    assert np.isclose(mean, sum(r**2))
    assert np.allclose(gradient, 2 * r)


def test_Displacement_fock_probabilities_jacobian_2_modes():
    r = tf.Variable([0.15, 0.2])

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r[0])
            pq.Q(1) | pq.Displacement(r=r[1])

        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [r])

    expected_fock_probabilities = np.exp(-sum(r**2)) * np.array(
        [
            1.0,
            r[0] ** 2,
            r[1] ** 2,
            r[0] ** 4 / 2,
            r[0] ** 2 * r[1] ** 2,
            r[1] ** 4 / 2,
        ]
    )

    expected_jacobian = np.exp(-sum(r**2)) * np.array(
        [
            -2 * r,
            [
                2 * r[0] * (1 - r[0] ** 2),
                -2 * r[1] * r[0] ** 2,
            ],
            [
                -2 * r[0] * r[1] ** 2,
                2 * r[1] * (1 - r[1] ** 2),
            ],
            [
                2 * r[0] ** 3 - r[0] ** 5,
                -2 * r[1] * r[0] ** 4 / 2,
            ],
            [
                r[1] ** 2 * 2 * r[0] * (1 - r[0] ** 2),
                r[0] ** 2 * 2 * r[1] * (1 - r[1] ** 2),
            ],
            [
                -2 * r[0] * r[1] ** 4 / 2,
                2 * r[1] ** 3 - r[1] ** 5,
            ],
        ]
    )

    assert np.allclose(
        fock_probabilities,
        expected_fock_probabilities,
        atol=1e-4,
    )

    assert np.allclose(
        jacobian,
        expected_jacobian,
        rtol=1e-2,
    )


def test_Squeezing_mean_photon_number_gradient_1_mode():
    r = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=8),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Squeezing(r=r)

        state = simulator.execute(program).state

        mean = state.mean_photon_number()

    gradient = tape.gradient(mean, [r])

    assert np.isclose(mean, np.sinh(r) ** 2)
    assert np.isclose(gradient, 2 * np.sinh(r) * np.cosh(r))


def test_Squeezing_mean_photon_number_gradient_1_mode_with_phaseshift():
    r = 0.1
    phi = tf.Variable(np.pi / 5)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=8),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Squeezing(r=r, phi=phi)

        state = simulator.execute(program).state

        _, variance = state.quadratures_mean_variance(modes=(0,))

    gradient = tape.jacobian(variance, [phi])

    assert np.isclose(variance, np.cosh(2 * r) - np.sinh(2 * r) * np.cos(phi))
    assert np.isclose(gradient, np.sinh(2 * r) * np.sin(phi))


def test_Beamsplitter_fock_probabilities_gradient_1_particle():
    theta = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=2),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta)

        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [theta])

    assert np.allclose(
        fock_probabilities,
        [0, np.cos(theta) ** 2, np.sin(theta) ** 2],
    )

    assert np.allclose(
        jacobian,
        [0, -2 * np.cos(theta) * np.sin(theta), 2 * np.cos(theta) * np.sin(theta)],
    )


def test_Beamsplitter_fock_probabilities_gradient_1_particle_with_phaseshift():
    theta = tf.Variable(np.pi / 3)
    phi = np.pi / 5

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=2),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta, phi=phi)

        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [theta])

    assert np.allclose(
        fock_probabilities,
        [0, np.cos(theta) ** 2, np.sin(theta) ** 2],
    )

    assert np.allclose(
        jacobian,
        [0, -2 * np.cos(theta) * np.sin(theta), 2 * np.cos(theta) * np.sin(theta)],
    )


def test_decomposed_Beamsplitter_state_vector_gradient_1_particle():
    theta = tf.Variable(11 * np.pi / 17)

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=2),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1, 0))

            pq.Q(1) | pq.Phaseshifter(-np.pi / 2)
            pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=0.0)
            pq.Q(0) | pq.Phaseshifter(theta)
            pq.Q(1) | pq.Phaseshifter(np.pi - theta)
            pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=0.0)
            pq.Q(1) | pq.Phaseshifter(-np.pi / 2)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [theta])

    assert np.allclose(
        state_vector,
        [0, np.cos(theta), np.sin(theta)],
    )

    assert np.allclose(
        jacobian,
        np.real([0, -np.sin(theta), np.cos(theta)]),
    )


def test_Beamsplitter_fock_probabilities_gradient_2_particles():
    theta = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(all) | pq.Beamsplitter(theta=theta)

        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [theta])

    assert np.allclose(
        fock_probabilities,
        [
            0,
            0,
            0,
            np.cos(theta) ** 4,
            2 * (np.cos(theta) * np.sin(theta)) ** 2,
            np.sin(theta) ** 4,
        ],
    )

    assert np.allclose(
        jacobian,
        [
            0,
            0,
            0,
            -4 * np.cos(theta) ** 3 * np.sin(theta),
            2 * (np.sin(2 * theta)) * np.cos(2 * theta),
            4 * np.sin(theta) ** 3 * np.cos(theta),
        ],
    )


def test_multiple_Beamsplitter_state_vector_gradient_2_particles():
    theta_1 = np.pi / 7
    phi_1 = 5 * np.pi / 9
    theta = tf.Variable(np.pi / 5)

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta_1, phi=phi_1)

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta, phi=0.0)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [theta])

    u00 = np.cos(theta_1) * np.cos(theta) - np.exp(1j * phi_1) * np.sin(
        theta_1
    ) * np.sin(theta)

    u10 = np.cos(theta_1) * np.sin(theta) + np.exp(1j * phi_1) * np.cos(theta) * np.sin(
        theta_1
    )

    u00_diff = -u10
    u10_diff = u00

    assert np.allclose(
        state_vector,
        [
            0,
            0,
            0,
            u00**2,
            np.sqrt(2) * u00 * u10,
            u10**2,
        ],
    )

    assert np.allclose(
        jacobian,
        np.real(
            [
                0,
                0,
                0,
                2 * u00 * u00_diff,
                np.sqrt(2) * (u00_diff * u10 + u00 * u10_diff),
                2 * u10 * u10_diff,
            ]
        ),
    )


def test_multiple_Beamsplitter_state_vector_gradient_2_particles_reversed():
    theta_1 = np.pi / 7
    phi_1 = 5 * np.pi / 9
    theta = tf.Variable(np.pi / 5)

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((2, 0))

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta, phi=0.0)

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta_1, phi=phi_1)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [theta])

    u00 = np.cos(theta_1) * np.cos(theta) - np.exp(-1j * phi_1) * np.sin(
        theta_1
    ) * np.sin(theta)

    u10 = np.cos(theta_1) * np.sin(theta) + np.exp(1j * phi_1) * np.cos(theta) * np.sin(
        theta_1
    )

    u00_diff = -u10.conj()
    u10_diff = u00.conj()

    assert np.allclose(
        state_vector,
        [
            0,
            0,
            0,
            u00**2,
            np.sqrt(2) * u00 * u10,
            u10**2,
        ],
    )

    assert np.allclose(
        jacobian,
        np.real(
            [
                0,
                0,
                0,
                2 * u00 * u00_diff,
                np.sqrt(2) * (u00_diff * u10 + u00 * u10_diff),
                2 * u10 * u10_diff,
            ]
        ),
    )


@pytest.mark.monkey
def test_jacobian_of_state_after_mixing_with_fix_Interferometer():
    c1 = tf.Variable(np.sqrt(0.2))
    c2 = tf.Variable(np.sqrt(0.3))
    c3 = tf.Variable(np.sqrt(0.5))

    d = 3

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=2),
        connector=pq.TensorflowConnector(),
    )

    U = unitary_group.rvs(3)

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1, 0, 0)) * c1
            pq.Q(all) | pq.StateVector((0, 1, 0)) * c2
            pq.Q(all) | pq.StateVector((0, 0, 1)) * c3

            pq.Q(all) | pq.Interferometer(U)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [c1, c2, c3])

    initial_state_vector = np.array([0.0, c1, c2, c3])

    fock_space_unitary = block_diag([1.0], U)

    assert np.allclose(state_vector, fock_space_unitary @ initial_state_vector)

    assert np.allclose(
        jacobian,
        np.real(
            [
                fock_space_unitary @ np.array([0, 1, 0, 0]),
                fock_space_unitary @ np.array([0, 0, 1, 0]),
                fock_space_unitary @ np.array([0, 0, 0, 1]),
            ]
        ),
    )


def test_Phaseshifter_density_matrix_gradient():
    phi = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=2), connector=pq.TensorflowConnector()
    )

    coefficients = [np.sqrt(0.6), np.sqrt(0.4)]

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(0) | pq.StateVector([0]) * coefficients[0]
            pq.Q(0) | pq.StateVector([1]) * coefficients[1]

            pq.Q(0) | pq.Phaseshifter(phi)

        state = simulator.execute(program).state

        density_matrix = state.density_matrix

    jacobian = tape.jacobian(density_matrix, [phi])

    assert np.allclose(
        density_matrix,
        [
            [
                coefficients[0] ** 2,
                coefficients[0] * coefficients[1] * np.exp(-1j * phi),
            ],
            [
                coefficients[0] * coefficients[1] * np.exp(1j * phi),
                coefficients[1] ** 2,
            ],
        ],
    )

    assert np.allclose(
        jacobian,
        [
            [0.0, -np.prod(coefficients) * np.sin(phi)],
            [-np.prod(coefficients) * np.sin(phi), 0.0],
        ],
    )


def test_Phaseshifter_density_matrix_gradient_is_zero_at_zero_phaseshift():
    phi = tf.Variable(0.0)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(0) | pq.StateVector([0]) * np.sqrt(0.5)
            pq.Q(0) | pq.StateVector([1]) * np.sqrt(0.3)
            pq.Q(0) | pq.StateVector([2]) * np.sqrt(0.2)

            pq.Q(0) | pq.Phaseshifter(phi)

        state = simulator.execute(program).state

        density_matrix = state.density_matrix

    jacobian = tape.jacobian(density_matrix, [phi])

    assert np.allclose(jacobian, np.zeros_like(jacobian))


def test_Interferometer_fock_probabilities():
    param = tf.Variable(0.0)

    config = pq.Config(cutoff=3)

    simulator = pq.PureFockSimulator(
        d=2, config=config, connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        hamiltonian = tf.stack(
            [
                tf.stack([0, tf.exp(param)]),
                tf.stack([tf.exp(param), 0]),
            ]
        )

        j = tf.complex(0.0, 1.0)
        interferometer = tf.linalg.expm(j * hamiltonian)
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1, 0)) / np.sqrt(2)
            pq.Q(all) | pq.StateVector((1, 1)) / np.sqrt(2)

            pq.Q(all) | pq.Interferometer(interferometer)

        state = simulator.execute(program).state

        density_matrix = state.density_matrix

    jacobian = tape.jacobian(density_matrix, [param])

    expected_jacobian = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.4546487, 0.0, 0.0, -0.3162077, 0.0],
            [0.0, 0.0, 0.4546488, -0.07391234, 0.0, -0.07391226],
            [0.0, 0.0, -0.07391234, -0.3784013, 0.0, -0.3784013],
            [0.0, -0.3162077, 0.0, 0.0, 0.75680256, 0.0],
            [0.0, 0.0, -0.07391226, -0.3784013, 0.0, -0.37840137],
        ],
    )

    assert np.allclose(jacobian, expected_jacobian)


def test_Squeezing2_mean_photon_number():
    r = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            pq.Q(all) | pq.Squeezing2(r=r)

        state = simulator.execute(program).state

        state.normalize()

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [r])

    assert np.allclose(jacobian, [-0.19349256, 0.0, 0.0, 0.0, 0.19349256, 0.0])


def test_Kerr_fock_probabilities_on_1_mode():
    xi = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=3), connector=pq.TensorflowConnector()
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0]) / np.sqrt(2)
        pq.Q(all) | pq.StateVector([2]) / np.sqrt(2)

        pq.Q(all) | pq.Kerr(xi=xi)

    with tf.GradientTape() as tape:
        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [xi])

    assert np.allclose(jacobian, [0.0, 0.0, 0.0], atol=1e-6)


def test_Kerr_density_matrix_on_1_mode():
    xi = tf.Variable(np.pi / 5)

    n = 2

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=n + 1), connector=pq.TensorflowConnector()
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0]) / np.sqrt(2)
        pq.Q(all) | pq.StateVector([n]) / np.sqrt(2)

        pq.Q(all) | pq.Kerr(xi=xi)

    with tf.GradientTape() as tape:
        state = simulator.execute(program).state

        density_matrix = state.density_matrix

    jacobian = tape.jacobian(density_matrix, [xi])

    coefficient = n**2

    gradient_of_0n_component = 1j * coefficient * np.exp(1j * xi * coefficient) / 2

    excepted_jacobian = [
        [0.0, 0.0, np.conj(gradient_of_0n_component)],
        [0.0, 0.0, 0.0],
        [gradient_of_0n_component, 0.0, 0.0],
    ]

    assert np.allclose(jacobian, np.real(excepted_jacobian))


def test_CubicPhase_fock_probabilities_on_1_mode():
    gamma = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=4), connector=pq.TensorflowConnector()
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0]) / np.sqrt(2)
        pq.Q(all) | pq.StateVector([2]) / np.sqrt(2)

        pq.Q(all) | pq.CubicPhase(gamma=gamma)

    with tf.GradientTape() as tape:
        state = simulator.execute(program).state

        fock_probabilities = state.fock_probabilities

    jacobian = tape.jacobian(fock_probabilities, [gamma])

    assert np.allclose(jacobian, [-0.16857366, 0.35571513, -0.5196387, 0.33249724])


def test_CrossKerr_density_matrix():
    xi = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=4), connector=pq.TensorflowConnector()
    )

    n = np.array(
        [
            [1, 1],
            [1, 2],
        ]
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q(0, 1) | pq.StateVector(n[0]) * np.sqrt(0.5)
            pq.Q(0, 1) | pq.StateVector(n[1]) * np.sqrt(0.5)

            pq.Q(all) | pq.CrossKerr(xi=xi)

        state = simulator.execute(program).state

        density_matrix = state.density_matrix

    jacobian = tape.jacobian(density_matrix, [xi])

    expected_jacobian = np.zeros_like(density_matrix)

    relative_phase_angle = n[0, 0] * n[0, 1] - n[1, 0] * n[1, 1]

    coefficient = 1j * relative_phase_angle * np.exp(1j * xi * relative_phase_angle)

    expected_jacobian[4, 8] = np.real(coefficient / 2)
    expected_jacobian[8, 4] = np.real(coefficient / 2)

    assert np.allclose(jacobian, expected_jacobian)


def test_mean_position_Displacement_gradient_on_1_mode():
    d = 1
    cutoff = 7

    alpha_ = tf.Variable(0.02)

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(r=alpha_)

    config = pq.Config(cutoff=cutoff)

    simulator = pq.PureFockSimulator(
        d=d, config=config, connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        state = simulator.execute(program).state
        mean = state.mean_position(mode=0)

    grad = tape.gradient(mean, [alpha_])

    assert np.allclose(mean, np.sqrt(2 * config.hbar) * alpha_)
    assert np.allclose(grad, np.sqrt(2 * config.hbar))


def test_mean_position_Displacement_and_Squeezing_gradient_on_1_mode():
    d = 1
    cutoff = 7

    alpha_ = tf.Variable(0.01)
    r_ = tf.Variable(0.05)

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(r=alpha_)
        pq.Q(all) | pq.Squeezing(r_)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff), connector=pq.TensorflowConnector()
    )

    with tf.GradientTape() as tape:
        state = simulator.execute(program).state
        mean = state.mean_position(mode=0)

    grad = tape.gradient(mean, [r_, alpha_])

    assert np.allclose(mean, 0.019024584947048066)
    assert np.allclose(grad, [-0.019024614, 1.9024585382680494])


def test_Displacement_state_vector_gradient():
    r = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [r])

    expected_state_vector = np.exp(-(r**2) / 2) * np.array([1, r, r**2 / np.sqrt(2)])
    expected_jacobian = np.array(
        [
            -r * np.exp(-(r**2) / 2),
            np.exp(-(r**2) / 2) * (1 - r**2),
            np.exp(-(r**2) / 2) * r * (2 - r**2) / np.sqrt(2),
        ]
    )
    assert np.allclose(state_vector, expected_state_vector)
    assert np.allclose(jacobian, expected_jacobian)


def test_complex_Displacement_state_vector_gradient():
    r = tf.Variable(0.1)
    phi = tf.Variable(np.pi / 3)

    alpha = tf.cast(r, tf.complex64) * tf.exp(1j * tf.cast(phi, tf.complex64))

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r, phi=phi)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [r, phi])

    expected_state_vector = np.exp(-(r**2) / 2) * np.array(
        [1, alpha, alpha**2 / np.sqrt(2)], dtype=complex
    )
    expected_jacobian = np.real(
        np.array(
            [
                [
                    -r * np.exp(-(r**2) / 2),
                    np.exp(-(r**2) / 2) * (1 - r**2) * np.exp(1j * phi),
                    np.exp(-(r**2) / 2)
                    * r
                    * (2 - r**2)
                    / np.sqrt(2)
                    * np.exp(1j * 2 * phi),
                ],
                [
                    0.0,
                    np.exp(-(r**2) / 2) * alpha * 1j,
                    np.exp(-(r**2) / 2) * alpha**2 / np.sqrt(2) * 2j,
                ],
            ]
        )
    )
    assert np.allclose(state_vector, expected_state_vector)
    assert np.allclose(jacobian, expected_jacobian)


def test_Squeezing_state_vector_gradient():
    r = tf.Variable(0.1)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Squeezing(r=r)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [r])

    expected_state_vector = (
        1 / np.sqrt(np.cosh(r)) * np.array([1, 0, -np.tanh(r) * np.sqrt(2) / 2])
    )
    expected_jacobian = np.array(
        [
            -np.sinh(r) / (2 * np.cosh(r) ** (3 / 2)),
            0,
            (3 * np.sinh(r) ** 2 - 2 * np.cosh(r) ** 2)
            / (2 * np.sqrt(2) * np.cosh(r) ** (5 / 2)),
        ]
    )
    assert np.allclose(state_vector, expected_state_vector)
    assert np.allclose(jacobian, expected_jacobian)


def test_complex_Squeezing_state_vector_gradient():
    r = tf.Variable(0.1)
    phi = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=3),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Squeezing(r=r, phi=phi)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [r, phi])

    expected_state_vector = (
        1
        / np.sqrt(np.cosh(r))
        * np.array([1, 0, -np.exp(1j * phi) * np.tanh(r) * np.sqrt(2) / 2])
    )
    expected_jacobian = np.real(
        np.array(
            [
                [
                    -np.sinh(r) / (2 * np.cosh(r) ** (3 / 2)),
                    0,
                    np.exp(1j * phi)
                    * (3 * np.sinh(r) ** 2 - 2 * np.cosh(r) ** 2)
                    / (2 * np.sqrt(2) * np.cosh(r) ** (5 / 2)),
                ],
                [
                    0,
                    0,
                    -1j
                    * np.exp(1j * phi)
                    * np.tanh(r)
                    * np.sqrt(2)
                    / (2 * np.sqrt(np.cosh(r))),
                ],
            ]
        )
    )

    assert np.allclose(state_vector, expected_state_vector)
    assert np.allclose(jacobian, expected_jacobian)


def test_displaced_state_Squeezing_state_vector_gradient():
    r = tf.Variable(0.1)
    phi = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=1,
        config=pq.Config(cutoff=5),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 5)

            pq.Q(0) | pq.Squeezing(r=r, phi=phi)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [r, phi])

    assert np.allclose(
        state_vector,
        [
            0.63519669 + 0.00657421j,
            0.50710742 + 0.37614499j,
            0.11111278 + 0.38482443j,
            -0.06637907 + 0.15485146j,
            -0.04802964 + 0.0180018j,
        ],
    )
    assert np.allclose(
        jacobian,
        [
            [0.27494016, 0.12359607, -0.23514226, 0.12369870, 0.53532110],
            [0.00687977, 0.02229462, 0.05786623, 0.07308305, 0.04782758],
        ],
    )


def test_displaced_state_Beamsplitter_state_vector_gradient():
    theta = tf.Variable(0.05)
    phi = tf.Variable(np.pi / 3)

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=6),
        connector=pq.TensorflowConnector(),
    )

    with tf.GradientTape() as tape:
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 5)
            pq.Q(1) | pq.Displacement(r=1.0, phi=np.pi / 5)

            pq.Q(0, 1) | pq.Beamsplitter(theta, phi)

        state = simulator.execute(program).state

        state_vector = state.state_vector

    jacobian = tape.jacobian(state_vector, [theta, phi])
    correct_state_vector = [
        0.36787944 + 0.0j,
        0.28045205 + 0.22344226j,
        0.29532688 + 0.23424946j,
        0.05521643 + 0.24089812j,
        0.08286355 + 0.3579549j,
        0.06217113 + 0.26594453j,
        -0.0601728 + 0.12539207j,
        -0.1090666 + 0.228548j,
        -0.11413312 + 0.24050364j,
        -0.06895407 + 0.14611768j,
        -0.06101656 + 0.02952233j,
        -0.12814978 + 0.06234706j,
        -0.16481668 + 0.08062809j,
        -0.14131579 + 0.06951122j,
        -0.07419819 + 0.03669684j,
        -0.02882161 - 0.0065087j,
        -0.06778149 - 0.01515268j,
        -0.10081657 - 0.02230856j,
        -0.10603167 - 0.02322176j,
        -0.07885371 - 0.01709068j,
        -0.03708827 - 0.00795441j,
    ]

    correct_jacobian = [
        [
            0.0,
            -0.35052943,
            -0.05328067,
            -0.49699682,
            -0.6256707,
            -0.3798101,
            -0.24836758,
            -0.54263073,
            -0.6643838,
            -0.45959768,
            0.02016149,
            -0.08133037,
            -0.26358867,
            -0.36225897,
            -0.26171276,
            0.10512505,
            0.17760125,
            0.16063823,
            0.06011391,
            -0.03620036,
            -0.05506517,
        ],
        [
            0.0,
            0.00747839,
            -0.01828559,
            -0.00636516,
            -0.01746456,
            -0.01902903,
            -0.01710664,
            -0.02221716,
            -0.01405889,
            -0.00294601,
            -0.01389679,
            -0.01826834,
            -0.00945533,
            0.00392901,
            0.00838149,
            -0.00578762,
            -0.00771638,
            -0.00271062,
            0.006368,
            0.01159076,
            0.00867541,
        ],
    ]

    assert np.allclose(state_vector, correct_state_vector)
    assert np.allclose(jacobian, correct_jacobian)
