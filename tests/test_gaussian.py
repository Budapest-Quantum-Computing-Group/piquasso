#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso import Program, Q, D, S
from piquasso.gaussian import GaussianState, GaussianBackend


class TestGaussian:
    @pytest.fixture(autouse=True)
    def setup(self):
        C = np.array(
            [
                [3, 1, 0],
                [1, 2, 0],
                [0, 0, 1],
            ],
            dtype=complex,
        )
        G = np.array(
            [
                [1, 3, 0],
                [3, 2, 0],
                [0, 0, 1],
            ],
            dtype=complex,
        )
        mean = np.ones(3, dtype=complex)

        state = GaussianState(C, G, mean)

        self.program = Program(
            state=state,
            backend_class=GaussianBackend
        )

    def test_displacement(self):
        alpha = 2 + 3j
        r = 4
        phi = np.pi/3

        with self.program:
            Q(0) | D(r=r, phi=phi)
            Q(1) | D(alpha=alpha)
            Q(2) | D(r=r, phi=phi) | D(alpha=alpha)

        self.program.execute()

        expected_mean = np.array(
            [
                1 + r * np.exp(1j * phi),
                1 + alpha,
                1 + alpha + r * np.exp(1j * phi),
            ],
            dtype=complex,
        )

        assert np.allclose(self.program.state.mean, expected_mean)

    def test_squeezing(self):
        r = 4
        phi = np.pi/3
        theta = 0.7
        amp = -0.6
        amp_1 = 0.8

        with self.program:
            Q(0) | D(r=r, phi=phi)
            Q(1) | S(amp=-0.6, theta=0.7)
            Q(2) | S(amp=0.8)

        self.program.execute()

        expected_mean = np.array(
            [
                1 + r * np.exp(1j * phi),
                np.cosh(amp) - np.exp(1j * theta) * np.sinh(amp),
                np.cosh(amp_1) - np.sinh(amp_1),
            ],
            dtype=complex,
        )
        expected_C = np.array(
            [
                [3., 2.64628377 + 1.23043049j, 0],
                [2.64628377 - 1.23043049j, 6.33563837, 0],
                [0, 0, 0.99062875],
            ],
            dtype=complex
        )

        expected_G = np.array(
            [
                [1, 4.04333517 + 0.4101435j, 0],
                [4.04333517 + 0.4101435j, 5.83468969 + 3.22991457j, 0],
                [0, 0, -0.98588746],
            ],
            dtype=complex
        )

        assert np.allclose(self.program.state.mean, expected_mean)
        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)
