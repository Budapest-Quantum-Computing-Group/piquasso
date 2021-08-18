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
from piquasso.api import constants
from piquasso.api.errors import InvalidParameter


def test_xp_representation(state, assets):
    assert np.allclose(
        assets.load("expected_xp_mean"),
        state.xp_mean,
    )
    assert np.allclose(
        assets.load("expected_xp_corr"),
        state.xp_corr,
    )
    assert np.allclose(
        assets.load("expected_xp_cov"),
        state.xp_cov,
    )


def test_quad_representation(state, assets):
    assert np.allclose(
        assets.load("expected_mean"),
        state.mean,
    )
    assert np.allclose(
        assets.load("expected_corr"),
        state.corr,
    )
    assert np.allclose(
        assets.load("expected_cov"),
        state.cov,
    )


def test_representation_roundtrip(state):
    initial_mean = state.mean
    initial_cov = state.cov

    state.mean = initial_mean
    state.cov = initial_cov

    final_mean = state.mean
    final_cov = state.cov

    assert np.allclose(final_mean, initial_mean)
    assert np.allclose(final_cov, initial_cov)


def test_representation_roundtrip_at_different_HBAR(state):
    constants.HBAR = 42

    initial_mean = state.mean
    initial_cov = state.cov

    state.mean = initial_mean
    state.cov = initial_cov

    final_mean = state.mean
    final_cov = state.cov

    assert np.allclose(final_mean, initial_mean)
    assert np.allclose(final_cov, initial_cov)


def test_wigner_function(state, assets):
    actual_result = state.wigner_function(
        positions=[[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]],
        momentums=[[2.0, 4.0, 6.0], [6.0, 8.0, 0.0]],
    )

    expected_result = assets.load("expected_wigner_function_result")

    assert np.allclose(expected_result, actual_result)


def test_reduced_rotated_mean_and_cov(state, assets):
    modes = (0, 2)
    phi = np.pi/2

    mean, cov = state.reduced_rotated_mean_and_cov(modes, phi)

    expected_mean = assets.load("expected_mean")
    expected_cov = assets.load("expected_cov")

    assert np.allclose(mean, expected_mean)
    assert np.allclose(cov, expected_cov)


class TestGaussianStateOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        with pq.Program() as program:
            pq.Q(all) | pq.Displacement(alpha=[1.0, 2.0, 3.0j])

            pq.Q(0, 1) | pq.Squeezing2(r=np.log(2.0), phi=0.0)

            pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi/2, phi=0)

        self.state = pq.GaussianState(d=3)
        self.state.apply(program)
        self.state.validate()

    def test_rotated(self):
        phi = np.pi / 2
        rotated_state = self.state.rotated(phi)

        expected_rotated_mean = np.array([0., 6.5, 0., -5.5, 6., 0.])
        expected_rotated_cov = np.array(
            [
                [ 4.25,    0.,  3.75,    0., 0., 0.],
                [   0.,  4.25,    0., -3.75, 0., 0.],
                [ 3.75,    0.,  4.25,   -0., 0., 0.],
                [   0., -3.75,   -0.,  4.25, 0., 0.],
                [   0.,    0.,    0.,    0., 2., 0.],
                [   0.,    0.,    0.,    0., 0., 2.],
            ]
        )

        assert np.allclose(
            expected_rotated_mean,
            rotated_state.mean,
        )
        assert np.allclose(
            expected_rotated_cov,
            rotated_state.cov,
        )

    def test_reduced(self):
        modes = (0, 2)

        reduced_state = self.state.reduced(modes)

        expected_reduced_mean = np.array([-6.5, 0., 0., 6.])
        expected_reduced_cov = np.array(
            [
                [4.25,   0., 0., 0.],
                [  0., 4.25, 0., 0.],
                [  0.,   0., 2., 0.],
                [  0.,   0., 0., 2.],
            ]
        )

        assert np.allclose(
            expected_reduced_mean,
            reduced_state.mean,
        )
        assert np.allclose(
            expected_reduced_cov,
            reduced_state.cov,
        )


class TestGaussianStateVacuum:
    def test_vacuum_covariance_is_proportional_to_identity(self):
        d = 2

        state = pq.GaussianState(d=d)

        expected_mean = np.zeros(2 * d)
        expected_covariance = np.identity(2 * d) * constants.HBAR

        assert np.allclose(state.mean, expected_mean)
        assert np.allclose(state.cov, expected_covariance)


def test_mean_and_covariance(state, assets):
    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)

    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(state.mean, expected_mean)
    assert np.allclose(state.cov, expected_cov)


def test_mean_and_covariance_with_different_HBAR(state, assets):
    constants.HBAR = 42

    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)
    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(state.mean, expected_mean)
    assert np.allclose(state.cov, expected_cov)


def test_mean_is_scaled_with_squared_HBAR(state, assets):
    scaling = 14

    mean_default_hbar = state.mean

    constants.HBAR *= scaling

    mean_different_hbar = state.mean

    assert np.allclose(mean_default_hbar * np.sqrt(scaling), mean_different_hbar)


def test_cov_is_scaled_with_HBAR(state):
    scaling = 14

    cov_default_hbar = state.cov

    constants.HBAR *= scaling

    cov_different_hbar = state.cov

    assert np.allclose(cov_default_hbar * scaling, cov_different_hbar)


def test_complex_covariance(state, assets):

    expected_complex_covariance = assets.load("expected_complex_covariance")

    assert np.allclose(state.complex_covariance, expected_complex_covariance)


def test_complex_covariance_does_not_scale_with_HBAR(state):
    complex_covariance_default_hbar = state.complex_covariance

    constants.HBAR = 42

    complex_covariance_different_hbar = state.complex_covariance

    assert np.allclose(
        complex_covariance_default_hbar,
        complex_covariance_different_hbar,
    )


def test_complex_covariance_transformed_directly_from_xp_cov(state):
    xp_cov = state.xp_cov
    d = len(xp_cov) // 2

    W = (1 / np.sqrt(2)) * np.block(
        [
            [np.identity(d), 1j * np.identity(d)],
            [np.identity(d), - 1j * np.identity(d)],
        ]
    )

    complex_covariance = W @ xp_cov @ W.conj().T / constants.HBAR

    assert np.allclose(complex_covariance, state.complex_covariance)


def test_complex_displacement(state, assets):
    expected_complex_displacement = assets.load("expected_complex_displacement")

    assert np.allclose(
        state.complex_displacement,
        expected_complex_displacement,
    )


def test_complex_displacement_do_not_scale_with_HBAR(state):
    complex_displacement_default_hbar = state.complex_displacement

    constants.HBAR = 42

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

    assert np.isclose(0., state.mean_photon_number((0, 1, 2)))
    assert np.isclose(0., state.mean_photon_number((0, 1)))
    assert np.isclose(0., state.mean_photon_number((0,)))


def test_mean_photon_number():
    alpha = 0.5 + 1j
    r = 1.
    phi = 3.
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(alpha=alpha)
        pq.Q(1) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Squeezing(r=r, phi=phi)
        pq.Q(2) | pq.Displacement(alpha=alpha)

    state = pq.GaussianState(d=3)
    state.apply(program)

    mean_photon_number_first_mode = np.abs(alpha)**2
    mean_photon_number_second_mode = np.sinh(r)**2
    total_mean_photon_number = 2 * (np.abs(alpha)**2 + np.sinh(r)**2)

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

    probability = state.get_particle_detection_probability(
        occupation_number=(0, 2)
    )

    assert np.isclose(probability, 0.0012355308401079989)
