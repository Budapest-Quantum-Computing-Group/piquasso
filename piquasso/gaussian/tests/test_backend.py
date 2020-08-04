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

    def test_phaseshift(self, tolerance):
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

        assert (np.abs(self.backend.state.C - expected_C) < tolerance).all()
        assert (np.abs(self.backend.state.G - expected_G) < tolerance).all()
        assert (
            np.abs(self.backend.state.mean - expected_mean) < tolerance
        ).all()
