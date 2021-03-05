#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso.gaussian.state import GaussianState
from piquasso.api import constants


class TestGaussianStateRepresentations:
    """
    TODO: Beware, the `m`, `C` and `G` are not realistic in a sense that they cannot
        be acquired by Gaussian transformations.
    """

    @pytest.fixture
    def m(self):
        return np.array([1 - 2j, 3 + 4j], dtype=complex)

    @pytest.fixture
    def C(self):
        return np.array(
            [
                [     1, 1 + 2j],
                [1 - 2j,      6],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def G(self):
        return np.array(
            [
                [3 + 1j, 1 + 1j],
                [1 + 1j, 2 - 3j],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def mu(self):
        return np.array([1, -2, 3, 4]) * np.sqrt(2 * constants.HBAR)

    @pytest.fixture
    def corr(self):
        return np.array(
            [
                [ 9,  2,  4,  6],
                [ 2, -3, -2,  0],
                [ 4, -2, 17, -6],
                [ 6,  0, -6,  9],
            ]
        ) * constants.HBAR

    @pytest.fixture
    def cov(self):
        return np.array(
            [
                [  5,  10,  -8, -10],
                [ 10, -19,  22,  32],
                [ -8,  22, -19, -54],
                [-10,  32, -54, -55],
            ]
        ) * constants.HBAR

    @pytest.fixture
    def xp_mean(self):
        return np.array([1, 3, -2, 4]) * np.sqrt(2 * constants.HBAR)

    @pytest.fixture
    def xp_corr(self):
        return np.array(
            [
                [ 9,  4,  2,  6],
                [ 4, 17, -2, -6],
                [ 2, -2, -3,  0],
                [ 6, -6,  0,  9]
            ]
        ) * constants.HBAR

    @pytest.fixture
    def xp_cov(self):
        return np.array(
            [
                [  5,  -8,  10, -10],
                [ -8, -19,  22, -54],
                [ 10,  22, -19,  32],
                [-10, -54,  32, -55],
            ]
        ) * constants.HBAR

    @pytest.fixture(autouse=True)
    def setup(self, C, G, m):
        self.state = GaussianState(C, G, m)

    def test_xp_representation(
        self,
        xp_mean,
        xp_corr,
        xp_cov,
    ):
        assert np.allclose(
            xp_mean,
            self.state.xp_mean,
        )
        assert np.allclose(
            xp_corr,
            self.state.xp_corr,
        )
        assert np.allclose(
            xp_cov,
            self.state.xp_cov,
        )

    def test_quad_representation(
        self,
        mu,
        corr,
        cov,
    ):
        assert np.allclose(
            mu,
            self.state.mu,
        )
        assert np.allclose(
            corr,
            self.state.corr,
        )
        assert np.allclose(
            cov,
            self.state.cov,
        )

    def test_wigner_function(self, mu, cov):
        quadrature_array = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]
        )

        expected_result = np.array(
            [
                0.00040656676635938727,
                0.01037619639200025,
            ]
        )

        actual_result = self.state.wigner_function(quadrature_array)

        assert np.allclose(expected_result, actual_result)


class TestGaussianStateOperations:
    @pytest.fixture
    def m(self):
        return np.array([1 - 2j, 3 + 4j, 2 - 5j], dtype=complex)

    @pytest.fixture
    def C(self):
        return np.array(
            [
                [     1, 1 + 2j, 4 - 6j],
                [1 - 2j,      6, 5 + 9j],
                [4 + 6j, 5 - 9j,    -10],
            ],
            dtype=complex,
        )

    @pytest.fixture
    def G(self):
        return np.array(
            [
                [3 + 1j, 1 + 1j, 2 + 2j],
                [1 + 1j, 2 - 3j, 5 - 6j],
                [2 + 2j, 5 - 6j,      7]
            ],
            dtype=complex,
        )

    @pytest.fixture(autouse=True)
    def setup(self, C, G, m):
        self.state = GaussianState(C, G, m)

    def test_rotated(self):
        phi = np.pi / 2
        rotated_state = self.state.rotated(phi)

        rotated_C = np.array(
            [
                [     1, 1 + 2j, 4 - 6j],
                [1 - 2j,      6, 5 + 9j],
                [4 + 6j, 5 - 9j,    -10],
            ]
        )

        rotated_G = np.array(
            [
                [-3 - 1j, -1 - 1j, -2 - 2j],
                [-1 - 1j, -2 + 3j, -5 + 6j],
                [-2 - 2j, -5 + 6j,      -7],
            ]
        )

        rotated_m = np.array([-2 - 1j,  4 - 3j, -5 - 2j])

        assert np.allclose(
            rotated_C,
            rotated_state.C,
        )
        assert np.allclose(
            rotated_G,
            rotated_state.G,
        )
        assert np.allclose(
            rotated_m,
            rotated_state.m,
        )

    def test_reduced(self):
        modes = (0, 2)

        reduced_state = self.state.reduced(modes)

        reduced_C = np.array(
            [
                [     1,   4 - 6j],
                [4 + 6j,      -10],
            ],
            dtype=complex
        )

        reduced_G = np.array(
            [
                [3 + 1j, 2 + 2j],
                [2 + 2j, 7 + 0j],
            ],
            dtype=complex
        )

        reduced_m = np.array([1 - 2j, 2 - 5j], dtype=complex)

        assert np.allclose(
            reduced_C,
            reduced_state.C,
        )
        assert np.allclose(
            reduced_G,
            reduced_state.G,
        )
        assert np.allclose(
            reduced_m,
            reduced_state.m,
        )

    def test_reduced_rotated_mean_and_cov(self):
        modes = (0, 2)
        phi = np.pi/2

        mean, cov = self.state.reduced_rotated_mean_and_cov(modes, phi)

        expected_mean = np.array([ -4,  -2, -10,  -4])

        expected_cov = np.array(
            [
                [ -38, -20,  -72, -64],
                [ -20,  10,  -24,   8],
                [ -72, -24, -266, -80],
                [ -64,   8,  -80, -42],
            ]
        )

        assert np.allclose(mean, expected_mean)
        assert np.allclose(cov, expected_cov)


class TestGaussianStateVacuum:
    def test_create_vacuum(self):
        number_of_modes = 3

        state = GaussianState.create_vacuum(d=number_of_modes)

        expected_m = np.zeros(number_of_modes, dtype=complex)
        expected_C = np.zeros((number_of_modes, number_of_modes), dtype=complex)
        expected_G = np.zeros((number_of_modes, number_of_modes), dtype=complex)

        assert np.allclose(state.m, expected_m)
        assert np.allclose(state.C, expected_C)
        assert np.allclose(state.G, expected_G)

    def test_vacuum_covariance_is_proportional_to_identity(self):
        number_of_modes = 2
        hbar = constants.HBAR

        state = GaussianState.create_vacuum(d=number_of_modes)

        expected_covariance = np.identity(2 * number_of_modes) * hbar

        assert np.allclose(state.cov, expected_covariance)
