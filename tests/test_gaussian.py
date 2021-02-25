#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso import Program, Q, D, S, P
from piquasso.gaussian import GaussianState


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
        m = np.ones(3, dtype=complex)

        state = GaussianState(C, G, m)

        self.program = Program(state=state)

    def test_displacement(self):
        alpha = 2 + 3j
        r = 4
        phi = np.pi/3

        with self.program:
            Q(0) | D(r=r, phi=phi)
            Q(1) | D(alpha=alpha)
            Q(2) | D(r=r, phi=phi) | D(alpha=alpha)

        self.program.execute()

        expected_m = np.array(
            [
                1 + r * np.exp(1j * phi),
                1 + alpha,
                1 + alpha + r * np.exp(1j * phi),
            ],
            dtype=complex,
        )

        assert np.allclose(self.program.state.m, expected_m)

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

        expected_m = np.array(
            [
                1 + r * np.exp(1j * phi),
                np.cosh(amp) - np.exp(1j * theta) * np.sinh(amp),
                np.cosh(amp_1) - np.sinh(amp_1),
            ],
            dtype=complex,
        )
        expected_C = np.array(
            [
                [23, 7.411872 - 3.74266246j, 0.89865793 - 1.55652119j],
                [7.411872 + 3.74266246j, 6.33563837, 0],
                [0.89865793 + 1.55652119j, 0, 0.99062875],
            ],
            dtype=complex
        )

        expected_G = np.array(
            [
                [-3 + 20.78460969j,  5.96736589 + 7.02381044j, 0.89865793 + 1.5565211j],
                [5.96736589 + 7.02381044j,  5.83468969 + 3.22991457j, 0],
                [0.89865793 + 1.55652119j,  0, -0.98588746]
            ],
            dtype=complex
        )

        assert np.allclose(self.program.state.m, expected_m)
        assert np.allclose(self.program.state.G, expected_G)
        assert np.allclose(self.program.state.C, expected_C)

    def test_quadratic_phase(self):
        with self.program:
            Q(0) | P(0)
            Q(1) | P(2)
            Q(2) | P(1)

        self.program.execute()

        expected_m = np.array(
            [
                1 + 0j, 1 + 2j, 1 + 1j
            ],
            dtype=complex,
        )
        expected_C = np.array(
            [
                [3 - 0j, 1 + 4j, 0 - 0j],
                [1 - 4j, 7 + 4j, 0 - 0j],
                [0 + 0j, 0 + 0j, 1.75 + 1j]
            ],
            dtype=complex,
        )

        expected_G = np.array(
            [
                [ 1 + 0j,  3 + 4j,  0 + 0j],
                [ 3 + 4j, -7 + 9j,  0 + 0j],
                [ 0 + 0j,  0 + 0j, -0.25 + 2.5j],
            ],
            dtype=complex
        )

        assert np.allclose(self.program.state.m, expected_m)
        assert np.allclose(self.program.state.G, expected_G)
        assert np.allclose(self.program.state.C, expected_C)

    def test_displacement_leaves_the_covariance_invariant(self):
        r = 1
        phi = 1

        initial_cov = self.program.state.cov

        with self.program:
            Q(0) | D(r=r, phi=phi)

        self.program.execute()

        final_cov = self.program.state.cov

        assert np.allclose(initial_cov, final_cov)


class TestTwoModeGaussian:
    def test_displacement_leaves_the_covariance_invariant(self):
        r = 1
        phi = 1

        C = np.array(
            [
                [       1.0, 0.1 - 0.2j],
                [0.1 - 0.2j,        1.0],
            ],
            dtype=complex,
        )

        G = np.array(
            [
                [1.0, 0.1],
                [0.1, 1.0]
            ],
            dtype=complex,
        )
        m = 3 * np.ones(2, dtype=complex)

        state = GaussianState(C, G, m)

        program = Program(state=state)

        initial_cov = program.state.cov

        with program:
            Q(0) | D(r=r, phi=phi)

        program.execute()

        final_cov = program.state.cov

        assert np.allclose(initial_cov, final_cov)

    def test_displaced_vacuum_stays_positive_definite(self):
        r = 1
        phi = 1

        C = np.array(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=complex,
        )

        G = np.array(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=complex,
        )
        m = np.zeros(2, dtype=complex)

        state = GaussianState(C, G, m)

        program = Program(state=state)

        with program:
            Q(0) | D(r=r, phi=phi)

        program.execute()

        assert np.all(np.linalg.eigvals(program.state.cov) > 0), (
            "The covariance of any Gaussan quantum state should be positive definite."
        )
