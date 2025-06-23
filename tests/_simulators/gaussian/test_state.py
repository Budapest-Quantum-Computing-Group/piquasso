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

from scipy.linalg import polar, coshm, sinhm, block_diag
from scipy.special import factorial2

import piquasso as pq
from piquasso.api.exceptions import InvalidParameter


def test_xxpp_representation(state, assets):
    assert np.allclose(
        assets.load("expected_xxpp_mean"),
        state.xxpp_mean_vector,
    )
    assert np.allclose(
        assets.load("expected_xxpp_correlation"),
        state.xxpp_correlation_matrix,
    )
    assert np.allclose(
        assets.load("expected_xxpp_covariance"),
        state.xxpp_covariance_matrix,
    )


def test_xpxp_representation(state, assets):
    assert np.allclose(
        assets.load("expected_xpxp_mean"),
        state.xpxp_mean_vector,
    )
    assert np.allclose(
        assets.load("expected_xpxp_correlation"),
        state.xpxp_correlation_matrix,
    )
    assert np.allclose(
        assets.load("expected_xpxp_covariance"),
        state.xpxp_covariance_matrix,
    )


def test_set_xpxp_mean_vector_invalid_shape_raises_InvalidState(state):
    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.xpxp_mean_vector = np.array([1, 2, 3])

    assert "Invalid 'mean' vector shape" in error.value.args[0]


def test_set_xpxp_mean_vector_invalid_shape_validate_False():
    state = pq.GaussianState(
        2, connector=pq.NumpyConnector(), config=pq.Config(validate=False)
    )

    state.xpxp_mean_vector = np.array([1, 2, 3])


def test_set_xpxp_covariance_matrix_invalid_shape_raises_InvalidState(state):
    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.xpxp_covariance_matrix = np.array([1, 2, 3])

    assert "Invalid 'cov' matrix shape" in error.value.args[0]


def test_set_xpxp_covariance_matrix_invalid_shape_validate_False():
    state = pq.GaussianState(
        2, connector=pq.NumpyConnector(), config=pq.Config(validate=False)
    )

    with pytest.raises(Exception) as exc:
        state.xpxp_covariance_matrix = np.array([1, 2, 3])

    assert not isinstance(exc, pq.api.exceptions.PiquassoException)


def test_representation_roundtrip(state):
    initial_mean_vector = state.xpxp_mean_vector
    initial_covariance_matrix = state.xpxp_covariance_matrix

    state.xpxp_mean_vector = initial_mean_vector
    state.xpxp_covariance_matrix = initial_covariance_matrix

    final_mean_vector = state.xpxp_mean_vector
    final_covariance_matrix = state.xpxp_covariance_matrix

    assert np.allclose(final_mean_vector, initial_mean_vector)
    assert np.allclose(final_covariance_matrix, initial_covariance_matrix)


def test_representation_roundtrip_at_different_HBAR(state):
    state._config.hbar = 42

    initial_mean_vector = state.xpxp_mean_vector
    initial_covariance_matrix = state.xpxp_covariance_matrix

    state.xpxp_mean_vector = initial_mean_vector
    state.xpxp_covariance_matrix = initial_covariance_matrix

    final_mean_vector = state.xpxp_mean_vector
    final_covariance_matrix = state.xpxp_covariance_matrix

    assert np.allclose(final_mean_vector, initial_mean_vector)
    assert np.allclose(final_covariance_matrix, initial_covariance_matrix)


def test_xp_representation_roundtrip(state):
    initial_mean = state.xxpp_mean_vector
    initial_cov = state.xxpp_covariance_matrix

    state.xxpp_mean_vector = initial_mean
    state.xxpp_covariance_matrix = initial_cov

    final_mean = state.xxpp_mean_vector
    final_cov = state.xxpp_covariance_matrix

    assert np.allclose(final_mean, initial_mean)
    assert np.allclose(final_cov, initial_cov)


def test_xp_representation_roundtrip_at_different_HBAR(state):
    state._config.hbar = 42

    initial_mean = state.xxpp_mean_vector
    initial_cov = state.xxpp_covariance_matrix

    state.xxpp_mean_vector = initial_mean
    state.xxpp_covariance_matrix = initial_cov

    final_mean = state.xxpp_mean_vector
    final_cov = state.xxpp_covariance_matrix

    assert np.allclose(final_mean, initial_mean)
    assert np.allclose(final_cov, initial_cov)


def test_wigner_function(state, assets):
    actual_result = state.wigner_function(
        positions=[[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]],
        momentums=[[2.0, 4.0, 6.0], [6.0, 8.0, 0.0]],
    )

    expected_result = assets.load("expected_wigner_function_result")
    assert np.allclose(expected_result, actual_result)


def test_wigner_plot_1D():
    # Create a Gaussian state
    config = pq.Config(cutoff=10, hbar=42)
    dim = 1
    with pq.Program() as initialization:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.10, phi=np.angle(1 - 0.5j))
        pq.Q(0) | pq.Squeezing(r=0.10)
    simulator = pq.GaussianSimulator(d=dim, config=config)
    state = simulator.execute(initialization).state
    state.validate()

    x = np.linspace(-5, 5, 20)
    p = np.linspace(-5, 5, 20)

    x = np.array([x[i : i + dim] for i in range(0, len(x), dim)])
    p = np.array([p[i : i + dim] for i in range(0, len(p), dim)])
    mode = 0
    try:
        state.plot_wigner(positions=x, momentums=p, mode=mode)
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")


def test_GaussState_plot_wigner_function_raises_InvalidModes_for_multiple_modes():
    alpha = 1 - 0.5j
    config = pq.Config(cutoff=10, hbar=42)
    dim = 1
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = pq.GaussianSimulator(d=dim, config=config)
    state = simulator.execute(program).state
    state.validate()

    with pytest.raises(pq.api.exceptions.InvalidModes):
        x = np.linspace(-5, 5, 20)
        p = np.linspace(-5, 5, 20)
        mode = 2
        state.plot_wigner(x, p, mode=mode)


def test_GaussState_plot_wigner_function_raises_InvalidModes_for_multiple_dimensions():
    alpha = 1 - 0.5j
    config = pq.Config(cutoff=10, hbar=42)
    dim = 2
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = pq.GaussianSimulator(d=dim, config=config)
    state = simulator.execute(program).state
    state.validate()

    with pytest.raises(pq.api.exceptions.InvalidModes):
        x = np.linspace(-5, 5, 20)
        p = np.linspace(-5, 5, 20)
        state.plot_wigner(x, p)


def test_reduced_rotated_mean_and_covariance(state, assets):
    modes = (0, 2)
    phi = np.pi / 2

    mean_vector, covariance_matrix = state.xpxp_reduced_rotated_mean_and_covariance(
        modes, phi
    )

    expected_mean = assets.load("expected_mean")
    expected_cov = assets.load("expected_covariance")

    assert np.allclose(mean_vector, expected_mean)
    assert np.allclose(covariance_matrix, expected_cov)


def test_rotated():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1.0)
        pq.Q(1) | pq.Displacement(r=2.0)
        pq.Q(2) | pq.Displacement(r=3.0, phi=np.pi / 2)

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state
    state.validate()

    phi = np.pi / 2
    rotated_state = state.rotated(phi)

    expected_rotated_mean_vector = np.array([0.0, 6.5, 0.0, -5.5, 6.0, 0.0])
    expected_rotated_covariance_matrix = np.array(
        [
            [4.25, 0.0, 3.75, 0.0, 0.0, 0.0],
            [0.0, 4.25, 0.0, -3.75, 0.0, 0.0],
            [3.75, 0.0, 4.25, -0.0, 0.0, 0.0],
            [0.0, -3.75, -0.0, 4.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )

    assert np.allclose(
        expected_rotated_mean_vector,
        rotated_state.xpxp_mean_vector,
    )
    assert np.allclose(
        expected_rotated_covariance_matrix,
        rotated_state.xpxp_covariance_matrix,
    )


def test_rotated_state_inherits_config():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1.0)
        pq.Q(1) | pq.Displacement(r=2.0)
        pq.Q(2) | pq.Displacement(r=3.0, phi=np.pi / 2)

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    config = pq.Config(hbar=1, seed_sequence=42)

    simulator = pq.GaussianSimulator(d=3, config=config)
    state = simulator.execute(program).state
    state.validate()

    reduced_state = state.rotated(phi=np.pi / 5)

    assert state._config == config
    assert reduced_state._config == state._config


def test_reduced():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1.0)
        pq.Q(1) | pq.Displacement(r=2.0)
        pq.Q(2) | pq.Displacement(r=3.0, phi=np.pi / 2)

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state
    state.validate()

    modes = (0, 2)

    reduced_state = state.reduced(modes)

    expected_reduced_mean_vector = np.array([-6.5, 0.0, 0.0, 6.0])
    expected_reduced_covariance_matrix = np.array(
        [
            [4.25, 0.0, 0.0, 0.0],
            [0.0, 4.25, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )

    assert np.allclose(
        expected_reduced_mean_vector,
        reduced_state.xpxp_mean_vector,
    )
    assert np.allclose(
        expected_reduced_covariance_matrix,
        reduced_state.xpxp_covariance_matrix,
    )


def test_reduced_state_inherits_config():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1.0)
        pq.Q(1) | pq.Displacement(r=2.0)
        pq.Q(2) | pq.Displacement(r=3.0, phi=np.pi / 2)

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    config = pq.Config(hbar=1, seed_sequence=42)

    simulator = pq.GaussianSimulator(d=3, config=config)
    state = simulator.execute(program).state
    state.validate()

    reduced_state = state.reduced(modes=(0,))

    assert state._config == config
    assert reduced_state._config == state._config


def test_GaussianState_get_marginal_fock_probabilities():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1.0)
        pq.Q(1) | pq.Displacement(r=2.0)
        pq.Q(2) | pq.Displacement(r=3.0, phi=np.pi / 2)

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    simulator = pq.GaussianSimulator(d=3, config=pq.Config(cutoff=5))
    state = simulator.execute(program).state
    state.validate()

    modes = (0, 2)
    expected_probabilities = state.reduced(modes).fock_probabilities
    actual_probabilities = state.get_marginal_fock_probabilities(modes)

    assert np.allclose(actual_probabilities, expected_probabilities)


def test_vacuum_covariance_is_proportional_to_identity():
    d = 2

    simulator = pq.GaussianSimulator(d=d)

    state = simulator.create_initial_state()

    expected_xpxp_mean = np.zeros(2 * d)
    expected_xpxp_covariance_matrix = np.identity(2 * d) * state._config.hbar

    assert np.allclose(state.xpxp_mean_vector, expected_xpxp_mean)
    assert np.allclose(state.xpxp_covariance_matrix, expected_xpxp_covariance_matrix)


def test_mean_and_covariance(state, assets):
    expected_mean_vector = assets.load("expected_mean") * np.sqrt(
        2 * state._config.hbar
    )

    expected_covariance_matrix = assets.load("expected_cov") * state._config.hbar

    assert np.allclose(state.xpxp_mean_vector, expected_mean_vector)
    assert np.allclose(state.xpxp_covariance_matrix, expected_covariance_matrix)


def test_mean_and_covariance_with_different_HBAR(state, assets):
    state._config.hbar = 42

    expected_mean_vector = assets.load("expected_mean") * np.sqrt(
        2 * state._config.hbar
    )
    expected_covariance_matrix = assets.load("expected_cov") * state._config.hbar

    assert np.allclose(state.xpxp_mean_vector, expected_mean_vector)
    assert np.allclose(state.xpxp_covariance_matrix, expected_covariance_matrix)


def test_mean_is_scaled_with_squared_HBAR(state, assets):
    scaling = 14

    mean_default_hbar = state.xpxp_mean_vector

    state._config.hbar *= scaling

    mean_different_hbar = state.xpxp_mean_vector

    assert np.allclose(mean_default_hbar * np.sqrt(scaling), mean_different_hbar)


def test_covariance_is_scaled_with_HBAR(state):
    scaling = 14

    cov_default_hbar = state.xpxp_covariance_matrix

    state._config.hbar *= scaling

    cov_different_hbar = state.xpxp_covariance_matrix

    assert np.allclose(cov_default_hbar * scaling, cov_different_hbar)


def test_complex_covariance(state, assets):
    expected_complex_covariance = assets.load("expected_complex_covariance")

    assert np.allclose(state.complex_covariance, expected_complex_covariance)


def test_complex_covariance_does_not_scale_with_HBAR(state):
    complex_covariance_default_hbar = state.complex_covariance

    state._config.hbar = 42

    complex_covariance_different_hbar = state.complex_covariance

    assert np.allclose(
        complex_covariance_default_hbar,
        complex_covariance_different_hbar,
    )


def test_complex_covariance_transformed_directly_from_xxpp_cov(state):
    xxpp_cov = state.xxpp_covariance_matrix
    d = len(xxpp_cov) // 2

    W = (1 / np.sqrt(2)) * np.block(
        [
            [np.identity(d), 1j * np.identity(d)],
            [np.identity(d), -1j * np.identity(d)],
        ]
    )

    complex_covariance = W @ xxpp_cov @ W.conj().T / state._config.hbar

    assert np.allclose(complex_covariance, state.complex_covariance)


def test_complex_displacement(state, assets):
    expected_complex_displacement = assets.load("expected_complex_displacement")

    assert np.allclose(
        state.complex_displacement,
        expected_complex_displacement,
    )


def test_complex_displacement_do_not_scale_with_HBAR(state):
    complex_displacement_default_hbar = state.complex_displacement

    state._config.hbar = 42

    complex_displacement_different_hbar = state.complex_displacement

    assert np.allclose(
        complex_displacement_default_hbar,
        complex_displacement_different_hbar,
    )


def test_quadratic_expectation_with_nonsymmetric_quadratic_coefficients(state):
    nonsymmetric_quadratic_coefficients = np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    any_vector = np.empty(2 * state.d)

    with pytest.raises(InvalidParameter):
        state.quadratic_polynomial_expectation(
            A=nonsymmetric_quadratic_coefficients,
            b=any_vector,
        )


def test_quadratic_expectation(state):
    rotation_angle = np.pi / 3
    constant_term = 100
    quadratic_coefficients = np.array(
        [
            [0, 1, 2, 0, 2, 0],
            [1, 0, 0, 1, 0, 1],
            [2, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )
    linear_coefficients = np.ones(2 * state.d)

    expectation_value = state.quadratic_polynomial_expectation(
        A=quadratic_coefficients,
        b=linear_coefficients,
        c=constant_term,
        phi=rotation_angle,
    )

    assert np.isclose(
        expectation_value,
        97.78556059428445,
    )


def test_mean_photon_number_vaccum():
    with pq.Program() as program:
        pass

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    assert np.isclose(0.0, state.mean_photon_number((0, 1, 2)))
    assert np.isclose(0.0, state.mean_photon_number((0, 1)))
    assert np.isclose(0.0, state.mean_photon_number((0,)))


def test_mean_photon_number():
    alpha = 0.5 + 1j
    r = 1.0
    phi = 3.0
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(1) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    mean_photon_number_first_mode = np.abs(alpha) ** 2
    mean_photon_number_second_mode = np.sinh(r) ** 2
    total_mean_photon_number = 2 * (np.abs(alpha) ** 2 + np.sinh(r) ** 2)

    assert np.isclose(mean_photon_number_first_mode, state.mean_photon_number((0,)))
    assert np.isclose(mean_photon_number_second_mode, state.mean_photon_number((1,)))
    assert np.isclose(total_mean_photon_number, state.mean_photon_number((0, 1, 2)))


def test_variance_photon_number():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=0.2, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.8, phi=np.pi / 2)
        pq.Q(2) | pq.Displacement(r=1.0)

    simulator = pq.GaussianSimulator(d=4)
    state = simulator.execute(program).state

    assert np.isclose(state.variance_photon_number((0,)), 0.04)
    assert np.isclose(
        state.variance_photon_number((1,)), 2 * np.sinh(0.8) ** 2 * np.cosh(0.8) ** 2
    )
    assert np.isclose(state.variance_photon_number((2,)), 1.0)
    assert np.isclose(state.variance_photon_number((3,)), 0.0)


def test_displaced_state_variance_photon_number_multimode():
    r_1 = 0.2
    r_2 = 0.3

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=r_1, phi=np.pi / 3)
        pq.Q(1) | pq.Displacement(r=r_2, phi=np.pi / 7)
        pq.Q(0, 1) | pq.Beamsplitter5050()

    simulator = pq.GaussianSimulator(d=4)
    state = simulator.execute(program).state

    assert np.isclose(state.variance_photon_number(), r_1**2 + r_2**2)


def test_GaussianState_get_particle_detection_probability():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability(occupation_number=(0, 2))

    assert np.isclose(probability, 0.0012355308401079989)


def test_GaussianState_threshold_and_particle_resolved_vacuum_detection_equivalence():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 7)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    vacuum_probability_1 = state.get_threshold_detection_probability(
        occupation_number=(0, 0)
    )
    vacuum_probability_2 = state.get_particle_detection_probability(
        occupation_number=(0, 0)
    )

    assert np.isclose(vacuum_probability_1, vacuum_probability_2)


def test_displaced_GaussianState_threshold_and_particle_resolved_vacuum_detection_equivalence():  # noqa: E501
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 7)
        pq.Q(1) | pq.Displacement(r=0.2, phi=-np.pi / 11)

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 7)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    vacuum_probability_1 = state.get_threshold_detection_probability(
        occupation_number=(0, 0)
    )
    vacuum_probability_2 = state.get_particle_detection_probability(
        occupation_number=(0, 0)
    )

    assert np.isclose(vacuum_probability_1, vacuum_probability_2)


def test_GaussianState_get_threshold_detection_probability_short_occupation_number():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.PiquassoException) as error:
        state.get_threshold_detection_probability(occupation_number=(1,))

    error_message = error.value.args[0]

    assert error_message == (
        "The specified occupation number should have length '2': "
        "occupation_number='(1,)'."
    )


def test_GaussianState_get_threshold_detection_probability_invalid_occupation_number():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.PiquassoException) as error:
        state.get_threshold_detection_probability(occupation_number=(1, 2))

    error_message = error.value.args[0]

    assert error_message == (
        "The specified occupation numbers must only contain '0' or '1': "
        "occupation_number='(1, 2)'."
    )


def test_GaussianState_get_threshold_detection_probability_2_mode():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    probability = state.get_threshold_detection_probability(occupation_number=(1, 0))

    assert np.isclose(probability, 0.0021696454384600313)


def test_displaced_GaussianState_get_threshold_detection_probability_2_mode():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 7)
        pq.Q(1) | pq.Displacement(r=0.2, phi=-np.pi / 11)

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    probability = state.get_threshold_detection_probability(occupation_number=(1, 0))

    assert np.isclose(probability, 0.012210371851308395)


def test_GaussianState_get_threshold_detection_probability_3_mode():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    probability = state.get_threshold_detection_probability(occupation_number=(1, 0, 1))

    assert np.isclose(probability, 0.00093484938797056)


def test_displaced_GaussianState_get_threshold_detection_probability_3_mode():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 7)
        pq.Q(1) | pq.Displacement(r=0.2, phi=-np.pi / 11)

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)
        pq.Q(1) | pq.Squeezing(r=0.05)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    probability = state.get_threshold_detection_probability(occupation_number=(1, 0, 1))

    assert np.isclose(probability, 0.0005136255432135231)


def test_GaussianState_get_threshold_detection_probability_sums_to_one_on_2_mode():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    probability00 = state.get_threshold_detection_probability(occupation_number=(0, 0))
    probability01 = state.get_threshold_detection_probability(occupation_number=(0, 1))
    probability10 = state.get_threshold_detection_probability(occupation_number=(1, 0))
    probability11 = state.get_threshold_detection_probability(occupation_number=(1, 1))

    assert np.isclose(
        probability00 + probability01 + probability10 + probability11, 1.0
    )


def test_displaced_GaussianState_get_threshold_detection_probability_sums_to_one_on_2_mode():  # noqa: E501
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 7)
        pq.Q(1) | pq.Displacement(r=0.2, phi=-np.pi / 11)

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = pq.GaussianSimulator(d=2)
    state = simulator.execute(program).state

    probability00 = state.get_threshold_detection_probability(occupation_number=(0, 0))
    probability01 = state.get_threshold_detection_probability(occupation_number=(0, 1))
    probability10 = state.get_threshold_detection_probability(occupation_number=(1, 0))
    probability11 = state.get_threshold_detection_probability(occupation_number=(1, 1))

    assert np.isclose(
        probability00 + probability01 + probability10 + probability11, 1.0
    )


@pytest.mark.monkey
def test_GaussianState_get_threshold_detection_probability_sums_to_one_random(
    generate_complex_symmetric_matrix, generate_unitary_matrix
):
    d = 3
    squeezing_matrix = generate_complex_symmetric_matrix(d)
    U, r = polar(squeezing_matrix)

    global_phase = generate_unitary_matrix(d)
    passive = global_phase @ coshm(r)
    active = global_phase @ sinhm(r) @ U.conj()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.GaussianTransform(passive=passive, active=active)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    binary_strings = []

    for i in range(2**d, 2 ** (d + 1)):
        binary_strings.append(tuple(int(x) for x in list(bin(i)[3:])))

    probabilities = []

    for binary_string in binary_strings:
        probabilities.append(
            state.get_threshold_detection_probability(occupation_number=binary_string)
        )

    assert np.isclose(sum(probabilities), 1.0)


@pytest.mark.monkey
def test_displaced_GaussianState_get_threshold_detection_probability_sums_to_one_random(
    generate_complex_symmetric_matrix, generate_unitary_matrix
):
    d = 3
    squeezing_matrix = generate_complex_symmetric_matrix(d)
    U, r = polar(squeezing_matrix)

    global_phase = generate_unitary_matrix(d)
    passive = global_phase @ coshm(r)
    active = global_phase @ sinhm(r) @ U.conj()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        for i in range(d):
            pq.Q(i) | pq.Displacement(
                r=np.random.rand(), phi=2 * np.pi * np.random.rand()
            )

        pq.Q() | pq.GaussianTransform(passive=passive, active=active)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    binary_strings = []

    for i in range(2**d, 2 ** (d + 1)):
        binary_strings.append(tuple(int(x) for x in list(bin(i)[3:])))

    probabilities = []

    for binary_string in binary_strings:
        probabilities.append(
            state.get_threshold_detection_probability(occupation_number=binary_string)
        )

    assert np.isclose(sum(probabilities), 1.0)


def test_GaussianState_fidelity():
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=2, phi=np.pi / 3)
        pq.Q(0) | pq.Displacement(r=0.1, phi=0)

    simulator = pq.GaussianSimulator(d=1)
    state_1 = simulator.execute(program_1).state

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=2, phi=-np.pi / 3)
        pq.Q(0) | pq.Displacement(r=-0.1, phi=0)

    state_2 = simulator.execute(program_2).state

    assert np.isclose(state_2.fidelity(state_1), 0.042150949)
    assert np.isclose(state_2.fidelity(state_1), state_1.fidelity(state_2))
    assert np.isclose(state_2.fidelity(state_2), 1.0)


def test_GaussianState_Vacuum_is_pure():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    assert state.is_pure()


def test_GaussianState_Vacuum_get_purity():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    assert np.isclose(state.get_purity(), 1.0)


def test_GaussianState_purify_vacuum():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state

    purification = state.purify()

    assert state.is_pure()
    assert purification.is_pure()
    assert purification.d == 2 * state.d
    assert purification.reduced(modes=(0, 1, 2)) == state


def test_GaussianState_purify_on_1_mode():
    with pq.Program() as program:
        pq.Q() | pq.Thermal([0.5])

        pq.Q(0) | pq.Squeezing(r=2, phi=np.pi / 3)
        pq.Q(0) | pq.Displacement(r=0.1, phi=0)

    simulator = pq.GaussianSimulator(d=1)
    mixed_state = simulator.execute(program).state

    purification = mixed_state.purify()

    assert not mixed_state.is_pure()
    assert purification.is_pure()
    assert purification.d == 2 * mixed_state.d
    assert purification.reduced(modes=(0,)) == mixed_state


def test_GaussianState_purify_on_2_modes():
    with pq.Program() as program:
        pq.Q() | pq.Thermal([0.5, 1.5])

        pq.Q(0) | pq.Squeezing(r=2, phi=np.pi / 3)
        pq.Q(0) | pq.Displacement(r=0.1, phi=0)
        pq.Q(1) | pq.Displacement(r=0.2, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    simulator = pq.GaussianSimulator(d=2)
    mixed_state = simulator.execute(program).state

    purification = mixed_state.purify()

    assert not mixed_state.is_pure()
    assert purification.is_pure()
    assert purification.d == 2 * mixed_state.d
    assert purification.reduced(modes=(0, 1)) == mixed_state


def test_GaussianState_purify_on_3_modes():
    with pq.Program() as program:
        pq.Q() | pq.Thermal([0.5, 1.5, 0.0])

        pq.Q(0) | pq.Squeezing(r=2, phi=np.pi / 3)
        pq.Q(0) | pq.Displacement(r=0.1, phi=0)
        pq.Q(1) | pq.Displacement(r=0.2, phi=np.pi / 3)
        pq.Q(2) | pq.Displacement(r=0.3, phi=-np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 7)

    simulator = pq.GaussianSimulator(d=3)
    mixed_state = simulator.execute(program).state

    purification = mixed_state.purify()

    assert not mixed_state.is_pure()
    assert purification.is_pure()
    assert purification.d == 2 * mixed_state.d
    assert purification.reduced(modes=(0, 1, 2)) == mixed_state


def test_density_matrix_Thermal_Interferometer(generate_unitary_matrix):
    d = 3

    preparation = pq.Program([pq.Thermal([0.1, 0.2, 0.3])])

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.Interferometer(U)

    simulator = pq.GaussianSimulator(d=d, config=pq.Config(cutoff=2))

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    fock_space_unitary = block_diag(np.array([1.0]), U)

    assert np.allclose(
        final_state.density_matrix,
        fock_space_unitary @ initial_state.density_matrix @ fock_space_unitary.conj().T,
    )


def test_get_purity_Thermal_Interferometer(generate_unitary_matrix):
    d = 3

    preparation = pq.Program([pq.Thermal([0.1, 0.2, 0.3])])

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.Interferometer(U)

    simulator = pq.GaussianSimulator(d=d, config=pq.Config(cutoff=7))

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    initial_density_matrix = initial_state.density_matrix
    final_density_matrix = final_state.density_matrix

    actual_initial_purity = np.trace(initial_density_matrix @ initial_density_matrix)
    actual_final_purity = np.trace(final_density_matrix @ final_density_matrix)

    expected_initial_purity = initial_state.get_purity()
    expected_final_purity = final_state.get_purity()

    assert np.isclose(actual_initial_purity, actual_final_purity)
    assert np.isclose(expected_initial_purity, expected_final_purity)
    assert np.isclose(expected_final_purity, actual_final_purity)


def test_get_xp_string_moment_vacuum_state_first_and_second_moments():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    state = pq.GaussianSimulator(d=1).execute(program).state

    assert np.isclose(state.get_xp_string_moment([]), 1.0)

    assert np.isclose(state.get_xp_string_moment([0]), 0.0)
    assert np.isclose(state.get_xp_string_moment([1]), 0.0)

    assert np.isclose(state.get_xp_string_moment([0, 0]), 1.0)
    assert np.isclose(state.get_xp_string_moment([1, 1]), 1.0)
    assert np.isclose(state.get_xp_string_moment([0, 1]), 1.0j)
    assert np.isclose(state.get_xp_string_moment([1, 0]), -1.0j)


def test_get_xp_string_moment_vacuum_energy():
    config = pq.Config()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    state = pq.GaussianSimulator(d=1, config=config).execute(program).state
    energy = (
        state.get_xp_string_moment([0, 0]) + state.get_xp_string_moment([1, 1])
    ) / 2

    assert np.isclose(energy, config.hbar / 2)


def test_get_xp_string_moment_vacuum_state_degree_4_decomposes_into_degree_2():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    state = pq.GaussianSimulator(d=1).execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0, 0, 1, 1]),
        state.get_xp_string_moment([0, 0]) * state.get_xp_string_moment([1, 1])
        + 2 * state.get_xp_string_moment([0, 1]) ** 2,
    )


def test_get_xp_string_moment_vacuum_state_against_explicit_formula():
    r"""
    We know that
    .. math::
        \bra{0} \hat{x}^n \ket{0} = \bra{0} \hat{p}^n \ket{0} = (n-1)!!,

    see, e.g., https://physics.stackexchange.com/a/748168/325715.
    """

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    state = pq.GaussianSimulator(d=1).execute(program).state

    assert np.isclose(state.get_xp_string_moment([0] * 6), factorial2(6 - 1))
    assert np.isclose(state.get_xp_string_moment([0] * 8), factorial2(8 - 1))

    assert np.isclose(state.get_xp_string_moment([1] * 6), factorial2(6 - 1))
    assert np.isclose(state.get_xp_string_moment([1] * 8), factorial2(8 - 1))


def test_get_xp_string_moment_admits_ccr_for_vacuum_state():
    config = pq.Config()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    state = pq.GaussianSimulator(d=1, config=config).execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0, 0, 1, 1])
        - state.get_xp_string_moment([0, 1, 0, 1]),
        1j * config.hbar * state.get_xp_string_moment([0, 1]),
    )


def test_get_xp_string_moment_single_mode_squeezed_state():
    r"""
    Consider :math:`\ket{r} := S(r) \ket{0}`, which is the state vector of a single-mode
    squeezed vacuum.

    Then, we know that
    .. math::
        \bra{r} \hat{X} \ket{r}
            = e^{-(n-m)r} \bra{0} \hat{X} \ket{0},

    where :math:`\hat{X}` is an XP-string consisting of :math:`n` :math:`\hat{x}`
    operators and :math:`m` :math:`\hat{p}` operators.
    """

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    vacuum_state = pq.GaussianSimulator(d=1).execute(program).state

    r = 0.2

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=r)

    state = pq.GaussianSimulator(d=1).execute(program).state

    xp_string = [0, 1, 0, 0, 1, 1, 0]

    m = np.count_nonzero(xp_string)
    n = len(xp_string) - m

    assert np.isclose(
        state.get_xp_string_moment(xp_string),
        np.exp(-(n - m) * r) * vacuum_state.get_xp_string_moment(xp_string),
    )


def test_get_xp_string_moment_coherent_state_first_and_second_moments():
    r"""
    This test uses the following fact:

    .. math::
        D(r, \phi)^\dagger \hat{x} D(r, \phi) = \hat{x} + \sqrt{2 \hbar} r \cos \phi, \\
        D(r, \phi)^\dagger \hat{p} D(r, \phi) = \hat{x} + \sqrt{2 \hbar} r \sin \phi.
    """

    hbar = 2
    config = pq.Config(hbar=hbar)

    r = 0.2
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r, phi=phi)

    state = pq.GaussianSimulator(d=1, config=config).execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0]), np.sqrt(2 * hbar) * r * np.cos(phi)
    )
    assert np.isclose(
        state.get_xp_string_moment([1]), np.sqrt(2 * hbar) * r * np.sin(phi)
    )
    assert np.isclose(
        state.get_xp_string_moment([0, 1]),
        hbar * (1j / 2 + 2 * r**2 * np.cos(phi) * np.sin(phi)),
    )

    assert np.isclose(
        state.get_xp_string_moment([0, 0]), hbar * (1 / 2 + 2 * r**2 * np.cos(phi) ** 2)
    )


def test_get_xp_string_moment_coherent_state_higher_moments():
    r"""
    This test uses the following fact:

    .. math::
        D(r, \phi)^\dagger \hat{x} D(r, \phi) &= \hat{x} + \sqrt{2 \hbar} r \cos \phi, \\
        D(r, \phi)^\dagger \hat{p} D(r, \phi) &= \hat{x} + \sqrt{2 \hbar} r \sin \phi.
    """  # noqa: E501

    hbar = 2
    config = pq.Config(hbar=hbar)

    r = 0.2
    phi = np.pi / 7

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r, phi=phi)

    state = pq.GaussianSimulator(d=1, config=config).execute(program).state

    x_moment = state.get_xp_string_moment([0])
    p_moment = state.get_xp_string_moment([1])

    moment_000 = state.get_xp_string_moment([0, 0, 0])

    assert np.isclose(
        moment_000,
        3 * (state.get_xp_string_moment([0, 0]) - x_moment**2) * x_moment + x_moment**3,
    )
    assert np.isclose(
        moment_000,
        np.sqrt(hbar) ** 3
        * (3 * r * np.cos(phi) / np.sqrt(2) + (np.sqrt(2) * r * np.cos(phi)) ** 3),
    )

    moment_011 = state.get_xp_string_moment([0, 1, 1])

    assert np.isclose(
        moment_011,
        x_moment * (state.get_xp_string_moment([1, 1]) - p_moment**2)
        + 2 * p_moment * (state.get_xp_string_moment([0, 1]) - x_moment * p_moment)
        + x_moment * p_moment**2,
    )

    assert np.isclose(
        moment_011,
        np.sqrt(hbar) ** 3
        * (
            np.sqrt(2**3) * r**3 * np.cos(phi) * np.sin(phi) ** 2
            + r * np.cos(phi) / np.sqrt(2)
            + 1j * np.sqrt(2) * r * np.sin(phi)
        ),
    )


def test_get_xp_string_moment_2_mode_first_and_second_moments():
    r"""
    This test uses the photonic `ControlledZ` gate (:math:`\hbar=1`):

    .. math::
        CZ(s)^\dagger \hat{x}_i CZ(s) &= \hat{x}_i, \\
        CZ(s)^\dagger \hat{p}_1 CZ(s) &= \hat{p}_1 + s \hat{x}_2, \\
        CZ(s)^\dagger \hat{p}_2 CZ(s) &= \hat{p}_2 + s \hat{x}_1,

    From these relations, the moments can be easily calculated.
    """
    config = pq.Config(hbar=1)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.ControlledZ(s=1)

    state = pq.GaussianSimulator(d=2, config=config).execute(program).state

    assert np.isclose(state.get_xp_string_moment([]), 1.0)
    assert np.isclose(state.get_xp_string_moment([0]), 0.0)
    assert np.isclose(state.get_xp_string_moment([1]), 0.0)
    assert np.isclose(state.get_xp_string_moment([2]), 0.0)
    assert np.isclose(state.get_xp_string_moment([3]), 0.0)

    second_moment_matrix = np.empty(shape=(4, 4), dtype=config.complex_dtype)

    for i in range(4):
        for j in range(4):
            second_moment_matrix[i, j] = state.get_xp_string_moment([i, j])

    assert np.allclose(
        second_moment_matrix,
        [
            [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.5j, 0.5 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.0 + 0.5j],
            [0.0 - 0.5j, 0.5 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.5 + 0.0j, 0.0 - 0.5j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
    )


def test_get_xp_string_moment_2_mode_higher_moments():
    r"""
    This test uses the photonic `ControlledZ` gate (:math:`\hbar=1`):

    .. math::
        CZ(s)^\dagger \hat{x}_i CZ(s) &= \hat{x}_i, \\
        CZ(s)^\dagger \hat{p}_1 CZ(s) &= \hat{p}_1 + s \hat{x}_2, \\
        CZ(s)^\dagger \hat{p}_2 CZ(s) &= \hat{p}_2 + s \hat{x}_1,

    From these relations, the moments can be easily calculated.
    """
    config = pq.Config(hbar=1)

    with pq.Program() as vacuum_program:
        pq.Q() | pq.Vacuum()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.ControlledZ(s=1)

    simulator = pq.GaussianSimulator(d=2, config=config)

    vacuum_state = simulator.execute(vacuum_program).state
    state = simulator.execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0, 2, 0, 2]),
        vacuum_state.get_xp_string_moment([0, 2, 0, 2])
        + vacuum_state.get_xp_string_moment([0, 1, 0, 2])
        + vacuum_state.get_xp_string_moment([0, 2, 0, 1])
        + vacuum_state.get_xp_string_moment([0, 1, 0, 1]),
    )


def test_get_xp_string_moment_admits_ccr_1():
    config = pq.Config()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Displacement(r=0.4)
        pq.Q(0) | pq.Squeezing(r=0.2, phi=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5)

    state = pq.GaussianSimulator(d=2, config=config).execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0, 0, 2, 2])
        - state.get_xp_string_moment([0, 2, 0, 2]),
        1j * config.hbar * state.get_xp_string_moment([0, 2]),
    )


def test_get_xp_string_moment_admits_ccr_2():
    config = pq.Config()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Displacement(r=0.4)
        pq.Q(0) | pq.Squeezing(r=0.2, phi=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5)

    state = pq.GaussianSimulator(d=2, config=config).execute(program).state

    assert np.isclose(
        state.get_xp_string_moment([0, 1, 2, 3]),
        state.get_xp_string_moment([0, 2, 1, 3]),
    )


def test_get_xp_string_moment_hbar_dependence():
    hbar_1 = 1.0
    hbar_2 = 3.0

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Displacement(r=0.4)
        pq.Q(0) | pq.Squeezing(r=0.2, phi=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5)

    state1 = (
        pq.GaussianSimulator(d=2, config=pq.Config(hbar=hbar_1)).execute(program).state
    )
    state2 = (
        pq.GaussianSimulator(d=2, config=pq.Config(hbar=hbar_2)).execute(program).state
    )

    indices = [0, 2, 1, 3, 0, 2]

    value1 = state1.get_xp_string_moment(indices)
    value2 = state2.get_xp_string_moment(indices)

    assert np.isclose(value1 * np.sqrt(hbar_2 / hbar_1) ** len(indices), value2)
