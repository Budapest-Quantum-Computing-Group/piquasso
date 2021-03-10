#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq

from piquasso.api.constants import HBAR


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

        self.program = pq.Program(
            state=pq.GaussianState(C, G, m),
        )

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
