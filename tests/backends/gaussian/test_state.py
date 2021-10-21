#
# Copyright 2021 Budapest Quantum Computing Group
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
from piquasso.api.errors import InvalidParameter


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
        pq.Q(all) | pq.Displacement(alpha=[1.0, 2.0, 3.0j])

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    state = pq.GaussianState(d=3)
    state.apply(program)
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


def test_reduced():
    with pq.Program() as program:
        pq.Q(all) | pq.Displacement(alpha=[1.0, 2.0, 3.0j])

        pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=0)

    state = pq.GaussianState(d=3)
    state.apply(program)
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


def test_vacuum_covariance_is_proportional_to_identity():
    d = 2

    state = pq.GaussianState(d=d)

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

    state = pq.GaussianState(d=3)
    state.apply(program)

    assert np.isclose(0.0, state.mean_photon_number((0, 1, 2)))
    assert np.isclose(0.0, state.mean_photon_number((0, 1)))
    assert np.isclose(0.0, state.mean_photon_number((0,)))


def test_mean_photon_number():
    alpha = 0.5 + 1j
    r = 1.0
    phi = 3.0
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(alpha=alpha)
        pq.Q(1) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Displacement(alpha=alpha)

    state = pq.GaussianState(d=3)
    state.apply(program)

    mean_photon_number_first_mode = np.abs(alpha) ** 2
    mean_photon_number_second_mode = np.sinh(r) ** 2
    total_mean_photon_number = 2 * (np.abs(alpha) ** 2 + np.sinh(r) ** 2)

    assert np.isclose(mean_photon_number_first_mode, state.mean_photon_number((0,)))
    assert np.isclose(mean_photon_number_second_mode, state.mean_photon_number((1,)))
    assert np.isclose(total_mean_photon_number, state.mean_photon_number((0, 1, 2)))


def test_GaussianState_get_particle_detection_probability():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    state = pq.GaussianState(d=2)
    state.apply(program)

    probability = state.get_particle_detection_probability(occupation_number=(0, 2))

    assert np.isclose(probability, 0.0012355308401079989)
