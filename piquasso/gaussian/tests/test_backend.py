#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from ..backend import GaussianBackend
from ..state import GaussianState


class TestGaussianBackend:
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

        self.backend = GaussianBackend(state=state)

    def test_phaseshift(self):
        angle = np.pi/3
        self.backend.phaseshift(params=(angle,), modes=(0,))

        phase = np.exp(1j * angle)

        expected_C = np.array(
            [
                [    3, np.conj(phase), 0],
                [phase,              2, 0],
                [    0,              0, 1],
            ],
            dtype=complex,
        )
        expected_G = np.array(
            [
                [-np.conj(phase), 3*phase, 0],
                [        3*phase,       2, 0],
                [              0,       0, 1],
            ],
            dtype=complex,
        )
        expected_mean = np.array([phase, 1, 1], dtype=complex)

        assert np.allclose(self.backend.state.C, expected_C)
        assert np.allclose(self.backend.state.G, expected_G)
        assert np.allclose(self.backend.state.mean, expected_mean)

    def test_beamsplitter(self):
        phi = np.pi/3
        theta = np.pi/4
        modes = 0, 1

        self.backend.beamsplitter(
            params=(phi, theta),
            modes=modes,
        )

        expected_C = np.array(
            [
                [             3, 0.5+0.8660254j, 0],
                [0.5-0.8660254j,              2, 0],
                [             0,              0, 1],
            ],
            dtype=complex,
        )
        expected_G = np.array(
            [
                [ 1.5+3.46410162j,  0.25+1.29903811j, 0],
                [0.25+1.29903811j, -0.75+2.16506351j, 0],
                [               0,                 0, 1],
            ],
            dtype=complex,
        )

        expected_mean = np.array(
            [
                1.06066017 + 0.61237244j,
                0.35355339 + 0.61237244j,
                1,
            ],
            dtype=complex,
        )

        assert np.allclose(self.backend.state.C, expected_C)
        assert np.allclose(self.backend.state.G, expected_G)
        assert np.allclose(self.backend.state.mean, expected_mean)

    def test_displacement_with_alpha(self):
        alpha = 3 - 4j

        self.backend.displacement(
            params=(alpha, ),
            modes=(1, ),
        )

        expected_mean = np.array(
            [
                1,
                1 + alpha,
                1,
            ],
            dtype=complex,
        )

        assert np.allclose(self.backend.state.mean, expected_mean)

    def test_displacement_with_r_and_phi(self):
        r = 5
        phi = np.pi / 4

        self.backend.displacement(
            params=(r, phi),
            modes=(1, ),
        )

        expected_mean = np.array(
            [
                1,
                1 + r * np.exp(1j * phi),
                1,
            ],
            dtype=complex,
        )

        assert np.allclose(self.backend.state.mean, expected_mean)
