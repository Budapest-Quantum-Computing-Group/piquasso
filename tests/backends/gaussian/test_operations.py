#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq

from piquasso.api import constants
from piquasso.api.constants import HBAR

from piquasso._math.linalg import is_selfadjoint, is_symmetric


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

        state = pq.GaussianState(C, G, m)

        self.program = pq.Program(state=state)

    def test_squeezing(self):
        amp = -0.6
        theta = 0.7

        with self.program:
            pq.Q(1) | pq.S(amp, theta)

        self.program.execute()

        expected_m = np.array(
            [
                1,
                np.cosh(amp) - np.exp(1j * theta) * np.sinh(amp),
                1,
            ],
            dtype=complex,
        )
        expected_C = np.array(
            [
                [3.0, 2.646283 + 1.23043j, 0.0],
                [2.646283 - 1.23043j, 6.335638, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
        expected_G = np.array(
            [
                [1.0, 4.043335 + 0.41014j, 0.0],
                [4.043335 + 0.41014j, 5.834689 + 3.229914j, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)
        assert np.allclose(self.program.state.m, expected_m)

    def test_phaseshift(self):
        angle = np.pi/3

        with self.program:
            pq.Q(0) | pq.R(phi=np.pi/3)

        self.program.execute()

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
        expected_m = np.array([phase, 1, 1], dtype=complex)

        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)
        assert np.allclose(self.program.state.m, expected_m)

    def test_beamsplitter(self):
        theta = np.pi/4
        phi = np.pi/3

        with self.program:
            pq.Q(0, 1) | pq.B(theta=theta, phi=phi)

        self.program.execute()

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

        expected_m = np.array(
            [
                1.06066017 + 0.61237244j,
                0.35355339 + 0.61237244j,
                1,
            ],
            dtype=complex,
        )

        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)
        assert np.allclose(self.program.state.m, expected_m)

    def test_displacement_with_alpha(self):
        alpha = 3 - 4j

        with self.program:
            pq.Q(1) | pq.D(alpha=alpha)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1 + alpha,
                    1,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [     1,  6 - 4j,      0],
                    [6 - 4j, 1 - 32j, 3 - 4j],
                    [     0,  3 - 4j,      1],
                ],
                dtype=complex,
            ),
            C=np.array(
                [
                    [      3,  4 - 4j,       0],
                    [ 4 + 4j,      33,  3 + 4j],
                    [      0,  3 - 4j,       1]
                ],
                dtype=complex,
            )
        )

        assert self.program.state == expected_state

    def test_displacement_with_r_and_phi(self):
        r = 5
        phi = np.pi / 4

        with self.program:
            pq.Q(1) | pq.D(r=r, phi=phi)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1 + r * np.exp(1j * phi),
                    1,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [1, 6.535534 + 3.535534j, 0],
                    [6.535534 + 3.535534j, 9.071068 + 32.071068j, 3.535534 + 3.535534j],
                    [0, 3.535534 + 3.535534j, 1]
                ],
                dtype=complex,
            ),
            C=np.array(
                [
                    [3,  4.535534 + 3.535534j, 0],
                    [4.535534 - 3.535534j, 34.071068, 3.535534 - 3.535534j],
                    [0,  3.535534 + 3.535534j, 1]
                ]
            )
        )

        assert self.program.state == expected_state

    def test_position_displacement(self):
        x = 4

        with self.program:
            pq.Q(1) | pq.X(x)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1 + x / np.sqrt(2 * HBAR),
                    1,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [ 1,  5,  0],
                    [ 5, 10,  2],
                    [ 0,  2,  1],
                ],
                dtype=complex,
            ),
            C=np.array(
                [
                    [ 3,  3, 0],
                    [ 3, 10, 2],
                    [ 0,  2, 1],
                ],
                dtype=complex,
            )
        )

        assert self.program.state == expected_state

    def test_momentum_displacement(self):
        p = 5

        with self.program:
            pq.Q(1) | pq.Z(p)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1 + 1j * p / np.sqrt(2 * HBAR),
                    1,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [1, 3 + 2.5j, 0],
                    [3 + 2.5j, -4.25 + 5j, 2.5j],
                    [0, 2.5j, 1],
                ],
                dtype=complex,
            ),
            C=np.array(
                [
                    [3, 1 + 2.5j, 0],
                    [1 - 2.5j, 8.25, - 2.5j],
                    [0, 2.5j, 1],
                ],
                dtype=complex,
            )
        )

        assert self.program.state == expected_state

    def test_displacement(self):
        alpha = 2 + 3j
        r = 4
        phi = np.pi/3

        with self.program:
            pq.Q(0) | pq.D(r=r, phi=phi)
            pq.Q(1) | pq.D(alpha=alpha)
            pq.Q(2) | pq.D(r=r, phi=phi) | pq.D(alpha=alpha)

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

    def test_displacement_and_squeezing(self):
        r = 4
        phi = np.pi/3
        theta = 0.7
        amp = -0.6
        amp_1 = 0.8

        with self.program:
            pq.Q(0) | pq.D(r=r, phi=phi)
            pq.Q(1) | pq.S(amp=-0.6, theta=0.7)
            pq.Q(2) | pq.S(amp=0.8)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1 + r * np.exp(1j * phi),
                    np.cosh(amp) - np.exp(1j * theta) * np.sinh(amp),
                    np.cosh(amp_1) - np.sinh(amp_1),
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [-3 + 20.7846097j,  5.9673657 + 7.0238104j, 0.8986579 + 1.556521j],
                    [5.9673659 + 7.0238104j,  5.8346897 + 3.2299146j, 0],
                    [0.8986579 + 1.5565212j,  0, -0.9858875]
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [23, 7.411872 - 3.74266246j, 0.89865793 - 1.55652119j],
                    [7.411872 + 3.74266246j, 6.33563837, 0],
                    [0.89865793 + 1.55652119j, 0, 0.99062875],
                ],
                dtype=complex
            )
        )

        assert self.program.state == expected_state

    def test_two_mode_squeezing(self):
        r = 4
        phi = np.pi/3

        with self.program:
            pq.Q(1, 2) | pq.S2(r=r, phi=phi)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    40.95319143 + 23.63376156j,
                    40.95319143 + 23.63376156j
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [1, 81.92469, 13.64495 + 23.633761j],
                    [81.92469, 1119.109 + 644.963396j, 1490.479 + 2581.5850j],
                    [13.64495 + 23.633761j, 1490.479 + 2581.5850j, 1 + 1289.926792j]
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [3, 27.308232, 40.93487 + 70.901284j],
                    [27.308232, 2980.95832, 1117.859119 + 645.396263j],
                    [40.93487 - 70.901284j, 1117.859119 - 645.396263j, 2979.95832],
                ],
                dtype=complex,
            ),
        )

        assert self.program.state == expected_state

    def test_controlled_X_gate(self):
        s = 2

        with self.program:
            pq.Q(1, 2) | pq.CX(s=s)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1,
                    3,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [ 1, 3,  4],
                    [ 3, 1,  5],
                    [ 4, 5, 10]
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [ 3,  1,  4],
                    [ 1,  3,  4],
                    [ 4,  4, 10]
                ],
                dtype=complex
            ),
        )

        assert self.program.state == expected_state

    def test_controlled_Z_gate(self):
        s = 2

        with self.program:
            pq.Q(1, 2) | pq.CZ(s=s)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    1 + 2j,
                    1 + 2j,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [  1,  3, 4j],
                    [  3, -3, 7j],
                    [ 4j, 7j, -8],
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [  3,   1, 4j],
                    [  1,   7, 2j],
                    [-4j, -2j, 10]
                ],
                dtype=complex
            ),
        )

        assert self.program.state == expected_state

    def test_fourier(self):
        with self.program:
            pq.Q(0) | pq.F()

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1j,
                    1,
                    1,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [-1, 3j, 0],
                    [3j, 2, 0],
                    [0, 0, 1],
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [3, -1j, 0],
                    [1j, 2, 0],
                    [0, 0, 1],
                ],
                dtype=complex
            ),
        )

        assert self.program.state == expected_state

    def test_mach_zehnder(self):
        int_ = np.pi/3
        ext = np.pi/4

        with self.program:
            pq.Q(1, 2) | pq.MZ(int_=int_, ext=ext)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1,
                    -0.9159756 + 0.87940952j,
                    -0.5865163 - 0.20886883j,
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [1, -1.4488887 + 0.3882286j, -2.5095489 + 0.672432j],
                    [-1.448889 + 0.388229j, 0.058013 - 0.899519j, 0.966506 - 0.058013j],
                    [-2.509549 + 0.672432j, 0.966506 - 0.058013j, 1.174038 - 0.966506j]
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [3, -0.48296291 + 0.129409523j, -0.8365163 + 0.224143868j],
                    [-0.48296291 - 0.129409523j,  1.25, 0.4330127],
                    [-0.8365163 - 0.224143868j, 0.4330127, 1.75]
                ],
                dtype=complex
            ),
        )

        assert self.program.state == expected_state

    def test_quadratic_phase(self):
        with self.program:
            pq.Q(0) | pq.P(0)
            pq.Q(1) | pq.P(2)
            pq.Q(2) | pq.P(1)

        self.program.execute()

        expected_state = pq.GaussianState(
            m=np.array(
                [
                    1 + 0j, 1 + 2j, 1 + 1j
                ],
                dtype=complex,
            ),
            G=np.array(
                [
                    [ 1 + 0j,  3 + 4j,  0 + 0j],
                    [ 3 + 4j, -7 + 9j,  0 + 0j],
                    [ 0 + 0j,  0 + 0j, -0.25 + 2.5j],
                ],
                dtype=complex
            ),
            C=np.array(
                [
                    [ 3 - 0j,  1 + 4j,    0 - 0j],
                    [ 1 - 4j, 11 - 0j,    0 - 0j],
                    [ 0 + 0j,  0 + 0j, 2.25 - 0j]
                ],
                dtype=complex,
            ),
        )

        assert self.program.state == expected_state

    def test_mean_and_covariance(self):
        with self.program:
            pq.Q(0) | pq.P(0)
            pq.Q(1) | pq.P(2)
            pq.Q(2) | pq.P(1)

        self.program.execute()

        expected_mean = np.array(
            [1, 0, 1, 2, 1, 1],
            dtype=complex,
        ) * np.sqrt(2 * HBAR)

        expected_cov = np.array(
            [
                [ 5,  0,  4,  8, -4, -4],
                [ 0,  5,  0, -4,  0,  0],
                [ 4,  0,  5, 10, -4, -4],
                [ 8, -4, 10, 21, -8, -8],
                [-4,  0, -4, -8,  1,  1],
                [-4,  0, -4, -8,  1,  2],
            ],
            dtype=complex
        ) * HBAR

        assert np.allclose(self.program.state.mu, expected_mean)
        assert np.allclose(self.program.state.cov, expected_cov)

    def test_mean_and_covariance_with_different_HBAR(self):
        with self.program:
            pq.Q(0) | pq.P(0)
            pq.Q(1) | pq.P(2)
            pq.Q(2) | pq.P(1)

        self.program.execute()

        constants.HBAR = 42

        expected_mean = np.array(
            [1, 0, 1, 2, 1, 1],
            dtype=complex,
        ) * np.sqrt(2 * constants.HBAR)

        expected_cov = np.array(
            [
                [ 5,  0,  4,  8, -4, -4],
                [ 0,  5,  0, -4,  0,  0],
                [ 4,  0,  5, 10, -4, -4],
                [ 8, -4, 10, 21, -8, -8],
                [-4,  0, -4, -8,  1,  1],
                [-4,  0, -4, -8,  1,  2],
            ],
            dtype=complex
        ) * constants.HBAR

        assert np.allclose(self.program.state.mu, expected_mean)
        assert np.allclose(self.program.state.cov, expected_cov)

        # TODO: We need to reset the value of HBAR. Create a better teardown for it!
        constants.HBAR = constants._HBAR_DEFAULT

    def test_displacement_leaves_the_covariance_invariant(self):
        r = 1
        phi = 1

        initial_cov = self.program.state.cov

        with self.program:
            pq.Q(0) | pq.D(r=r, phi=phi)

        self.program.execute()

        final_cov = self.program.state.cov

        assert np.allclose(initial_cov, final_cov)


class TestDisplacementGate:
    def test_displacement_leaves_the_covariance_invariant(self):
        r = 1
        phi = 1

        C = np.array(
            [
                [       1.0, 0.1 - 0.2j],
                [0.1 + 0.2j,        1.0],
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

        state = pq.GaussianState(C, G, m)

        program = pq.Program(state=state)

        initial_cov = program.state.cov

        with program:
            pq.Q(0) | pq.D(r=r, phi=phi)

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

        state = pq.GaussianState(C, G, m)

        program = pq.Program(state=state)

        with program:
            pq.Q(0) | pq.D(r=r, phi=phi)

        program.execute()

        assert np.all(np.linalg.eigvals(program.state.cov) > 0), (
            "The covariance of any Gaussan quantum state should be positive definite."
        )

    def test_multiple_displacements_leave_the_covariance_invariant(self):
        vacuum = pq.GaussianState.create_vacuum(d=3)

        initial_cov = vacuum.cov

        with pq.Program() as program:
            pq.Q() | vacuum

            pq.Q(0) | pq.D(r=2, phi=np.pi/3)
            pq.Q(1) | pq.D(r=1, phi=np.pi/4)
            pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        program.execute()

        assert np.allclose(program.state.cov, initial_cov)


class TestGaussianPassiveTransform:

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
        self.program = pq.Program(
            state=pq.GaussianState(C, G, m)
        )

    def test_apply_passive_to_C_and_G_for_1_modes(self, C, G):
        alpha = np.exp(1j * np.pi/3)

        T = np.array(
            [
                [alpha]
            ],
            dtype=complex,
        )

        expected_C = C.copy()

        expected_C[1, 1] = alpha.conjugate() * C[1, 1] * alpha

        expected_C[1, 0] = alpha.conjugate() * C[1, 0]
        expected_C[1, 2] = alpha.conjugate() * C[1, 2]

        expected_C[0, 1] = C[0, 1] * alpha
        expected_C[2, 1] = C[2, 1] * alpha

        expected_G = G.copy()

        expected_G[1, 1] = alpha * G[1, 1] * alpha

        expected_G[1, 0] = alpha * G[1, 0]
        expected_G[1, 2] = alpha * G[1, 2]

        expected_G[0, 1] = G[0, 1] * alpha
        expected_G[2, 1] = G[2, 1] * alpha

        with self.program:
            pq.Q(1) | pq.PassiveTransform(T)

        self.program.execute()

        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)

    def test_apply_passive_to_C_and_G_for_2_modes(self, C, G):

        T = np.array(
            [
                [     1, 5 + 6j],
                [5 - 6j,      7],
            ],
            dtype=complex,
        )

        expected_C = np.zeros((3, 3), dtype=complex)

        expected_C[:2, :2] = T.conjugate() @ C[:2, :2] @ T.transpose()

        expected_C[(0, 1), 2] = T.conjugate() @ C[(0, 1), 2]
        expected_C[2, (0, 1)] = C[2, (0, 1)] @ T.transpose()

        expected_C[2, 2] = C[2, 2]

        expected_G = np.zeros((3, 3), dtype=complex)

        expected_G[:2, :2] = T @ G[:2, :2] @ T.transpose()

        expected_G[(0, 1), 2] = T @ G[(0, 1), 2]
        expected_G[2, (0, 1)] = G[2, (0, 1)] @ T.transpose()

        expected_G[2, 2] = G[2, 2]

        with self.program:
            pq.Q(0, 1) | pq.PassiveTransform(T)

        self.program.execute()

        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)

    def test_apply_passive_to_C_and_G_for_all_modes(self, C, G):
        T = np.array(
            [
                [       1,    3 + 4j,   9 - 10j],
                [  3 - 4j,         7, -11 + 12j],
                [ 9 + 10j, -11 - 12j,        18]
            ],
            dtype=complex,
        )

        expected_C = T.conjugate() @ C @ T.transpose()

        expected_G = T @ G @ T.transpose()

        with self.program:
            pq.Q(0, 1, 2) | pq.PassiveTransform(T)

        self.program.execute()

        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)


class TestGaussianTransform:

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
        self.program = pq.Program(
            state=pq.GaussianState(C, G, m)
        )

    def test_apply_gaussian_transform_for_1_modes(self, C, G):
        alpha = np.exp(1j * np.pi/3)

        beta = np.exp(1j * np.pi/4)

        passive_transform = np.array(
            [
                [alpha]
            ],
            dtype=complex,
        )

        active_transform = np.array(
            [
                [beta]
            ],
            dtype=complex,
        )

        with self.program:
            pq.Q(1) | pq.GaussianTransform(P=passive_transform, A=active_transform)

        self.program.execute()

        expected_m = np.array(
            [1 - 2j, 2.98564585 + 3.89096943j, 2 - 5j],
        )

        expected_C = np.array(
            [
                [1, 0.18216275 + 1.8660254j, 4 - 6j],
                [0.182162 - 1.86602j, 18.41661758, 9.587121 - 7.608301j],
                [4 + 6j, 9.58712185 + 7.60830161j, -10]
            ]
        )

        expected_G = np.array(
            [
                [3 + 1j, 1.75529494 + 0.65891862j, 2 + 2j],
                [1.755294 + 0.658918j, -4.766571 + 17.789086j, 4.86772 + 11.229621j],
                [2 + 2j, 4.8677253 + 11.22962196j, 7 + 0j]
            ]
        )

        assert np.allclose(self.program.state.m, expected_m)
        assert np.allclose(self.program.state.C, expected_C)
        assert np.allclose(self.program.state.G, expected_G)


def test_apply_passive(
    generate_complex_symmetric_matrix,
    generate_hermitian_matrix,
    generate_unitary_matrix
):
    C = generate_hermitian_matrix(5)
    G = generate_complex_symmetric_matrix(5)
    m = np.random.rand(5) + 1j * np.random.rand(5)

    program = pq.Program(state=pq.GaussianState(C=C, G=G, m=m))

    expected_m = m.copy()
    expected_C = C.copy()
    expected_G = G.copy()

    T = generate_unitary_matrix(3)

    expected_m[(0, 1, 3), ] = T @ expected_m[(0, 1, 3), ]

    columns = np.array(
        [
            [0, 1, 3],
            [0, 1, 3],
            [0, 1, 3],
        ]
    )

    rows = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
        ]
    )

    index = rows, columns

    expected_C[index] = T.conjugate() @ expected_C[index] @ T.transpose()
    expected_C[(0, 1, 3), 2] = T.conjugate() @ expected_C[(0, 1, 3), 2]
    expected_C[(0, 1, 3), 4] = T.conjugate() @ expected_C[(0, 1, 3), 4]
    expected_C[:, (0, 1, 3)] = np.conj(expected_C[(0, 1, 3), :]).transpose()

    expected_G[index] = T @ expected_G[index] @ T.transpose()
    expected_G[(0, 1, 3), 2] = T @ expected_G[(0, 1, 3), 2]
    expected_G[(0, 1, 3), 4] = T @ expected_G[(0, 1, 3), 4]
    expected_G[:, (0, 1, 3)] = expected_G[(0, 1, 3), :].transpose()

    with program:
        pq.Q(0, 1, 3) | pq.PassiveTransform(T)

    program.execute()

    assert np.allclose(program.state.m, expected_m)
    assert np.allclose(program.state.C, expected_C)
    assert np.allclose(program.state.G, expected_G)


def test_complex_circuit():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=5)

        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        pq.Q(0, 1) | pq.B(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504, 3.289367083610399)

        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        pq.Q(0, 1) | pq.B(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504, 3.289367083610399)

    program.execute()

    assert is_selfadjoint(program.state.C)
    assert is_symmetric(program.state.G)
