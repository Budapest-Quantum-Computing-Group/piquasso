#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso.gaussian.state import GaussianState
from piquasso import constants


class TestGaussianState:

    @pytest.fixture
    def mean(self):
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
        return np.array([1, -2, 3, 4]) / np.sqrt(2 * constants.HBAR_DEFAULT)

    @pytest.fixture
    def corr(self):
        return np.array(
            [
                [ 9,  2,  4,  6],
                [ 2,  3, -2,  0],
                [ 4, -2, 17, -6],
                [ 6,  0, -6, -9],
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture
    def xp_mean(self):
        return np.array([1, 3, -2, 4]) / np.sqrt(2 * constants.HBAR_DEFAULT)

    @pytest.fixture
    def xp_corr(self):
        return np.array(
            [
                [ 9,  4,  2,  6],
                [ 4, 17, -2, -6],
                [ 2, -2,  3,  0],
                [ 6, -6,  0, -9]
            ]
        ) * constants.HBAR_DEFAULT

    @pytest.fixture(autouse=True)
    def setup(self, C, G, mean):
        self.state = GaussianState(C, G, mean)

    def test_xp_representation(
        self,
        xp_mean,
        xp_corr,
    ):
        assert np.allclose(
            xp_mean,
            self.state.xp_mean,
        )
        assert np.allclose(
            xp_corr,
            self.state.xp_corr,
        )

    def test_quad_representation(
        self,
        mu,
        corr,
    ):
        assert np.allclose(
            mu,
            self.state.mu,
        )
        assert np.allclose(
            corr,
            self.state.corr,
        )
