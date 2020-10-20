#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso import Program, Q, D
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
            Q(0) | D(r, phi)
            Q(1) | D(alpha)
            Q(2) | D(r, phi) | D(alpha)

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
