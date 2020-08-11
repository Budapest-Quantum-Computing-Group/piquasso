#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

from ..backend import PassiveGaussianBackend
from ..state import PassiveGaussianState


@pytest.fixture(scope="session")
def tolerance():
    return 10E-8


class TestPassiveGaussianBackend:
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

        state = PassiveGaussianState(C)
        self.backend = PassiveGaussianBackend(state=state)

    def test_phaseshift(self, tolerance):
        angle = np.pi / 3
        k = 0

        self.backend.phaseshift((angle,), (k,))

        phase = np.exp(1j * angle)

        expected_C = np.array(
            [
                [3, np.conj(phase), 0],
                [phase, 2, 0],
                [0, 0, 1],
            ],
            dtype=complex,
        )

        assert np.allclose(self.backend.state.C, expected_C, tolerance)

    def test_beamsplitter(self, tolerance):
        phi = np.pi / 3
        theta = np.pi / 4
        modes = 0, 1

        self.backend.beamsplitter((phi, theta), modes)

        expected_C = np.array(
            [
                [2.86237244 - 0.j, - 0.05618622 + 1.05618622j, 0. + 0.j],
                [-0.05618622 - 1.05618622j, 2.13762756 - 0.j, 0. + 0.j],
                [0. - 0.j, 0. - 0.j, 1. + 0.j]
            ],
            dtype=complex,
        )

        assert np.allclose(self.backend.state.C, expected_C, tolerance)
