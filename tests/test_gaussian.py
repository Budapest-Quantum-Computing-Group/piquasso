#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq

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
            state=GaussianState(C, G, m)
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
            state=GaussianState(C, G, m)
        )

    def test_apply_transform_for_1_modes(self, C, G):
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
                [0.182162 - 1.86602j, 9.416617 + 5.196152j, 9.587121 - 7.608301j],
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

    program = pq.Program(state=GaussianState(C=C, G=G, m=m))

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
