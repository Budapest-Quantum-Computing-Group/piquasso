#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq
from piquasso.api import constants


@pytest.fixture
def state(program):
    return program.state


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
    m = state.m
    G = state.G
    C = state.C

    mean = state.mean
    cov = state.cov

    state._apply_mean_and_cov(mean=mean, cov=cov)

    assert np.allclose(state.m, m)
    assert np.allclose(state.G, G)
    assert np.allclose(state.C, C)


def test_wigner_function(state, assets):
    quadrature_array = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [5, 6, 7, 8, 9, 0],
        ]
    )

    actual_result = state.wigner_function(quadrature_array)

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
    """
    NOTE: The m, C, G matrices are not real in this test class.
    """

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
        self.state = pq.GaussianState(C, G, m)

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


class TestGaussianStateVacuum:
    def test_create_vacuum(self):
        number_of_modes = 3

        state = pq.GaussianState.create_vacuum(d=number_of_modes)

        expected_m = np.zeros(number_of_modes, dtype=complex)
        expected_C = np.zeros((number_of_modes, number_of_modes), dtype=complex)
        expected_G = np.zeros((number_of_modes, number_of_modes), dtype=complex)

        assert np.allclose(state.m, expected_m)
        assert np.allclose(state.C, expected_C)
        assert np.allclose(state.G, expected_G)

    def test_vacuum_covariance_is_proportional_to_identity(self):
        number_of_modes = 2
        hbar = constants.HBAR

        state = pq.GaussianState.create_vacuum(d=number_of_modes)

        expected_covariance = np.identity(2 * number_of_modes) * hbar

        assert np.allclose(state.cov, expected_covariance)


def test_mean_and_covariance(program, assets):
    program.execute()
    program.state.validate()

    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)

    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(program.state.mean, expected_mean)
    assert np.allclose(program.state.cov, expected_cov)


def test_mean_and_covariance_with_different_HBAR(program, assets):
    program.execute()
    program.state.validate()

    constants.HBAR = 42

    expected_mean = assets.load("expected_mean") * np.sqrt(2 * constants.HBAR)
    expected_cov = assets.load("expected_cov") * constants.HBAR

    assert np.allclose(program.state.mean, expected_mean)
    assert np.allclose(program.state.cov, expected_cov)

    # TODO: We need to reset the value of HBAR. Create a better teardown for it!
    constants.HBAR = constants._HBAR_DEFAULT
