#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

from piquasso import Program, Q
from piquasso.operations import B, Interferometer, R, Sampling
from ..state import SamplingState


class TestSamplingBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        initial_state = SamplingState(1, 1, 1, 0, 0)
        self.program = Program(initial_state)

    def test_program(self):
        U = np.array([
            [.5, 0, 0],
            [0, .5j, 0],
            [0, 0, -1]
        ], dtype=complex)
        with self.program:
            Q(0, 1) | B(.5)
            Q(1, 2, 3) | Interferometer(U)
            Q(3) | R(.5)
            Sampling(shots=10)

    def test_interferometer(self):
        U = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=complex)
        with self.program:
            Q(4, 3, 1) | Interferometer(U)
        self.program.execute()

        expected_interferometer = np.array([
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ], dtype=complex)

        assert np.allclose(self.program.state.interferometer, expected_interferometer)

    def test_phaseshifter(self):
        phi = np.pi / 2
        with self.program:
            Q(2) | R(phi)
        self.program.execute()

        x = np.exp(1j * phi)
        expected_interferometer = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, x, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=complex)

        assert np.allclose(self.program.state.interferometer, expected_interferometer)

    def test_beamsplitter(self):
        theta = np.pi / 4
        phi = np.pi / 3
        with self.program:
            Q(1, 3) | B(theta, phi)
        self.program.execute()

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)
        rc = np.conj(r)
        expected_interferometer = np.array([
            [1, 0, 0, 0, 0],
            [0, t, 0, rc, 0],
            [0, 0, 1, 0, 0],
            [0, -r, 0, t, 0],
            [0, 0, 0, 0, 1],
        ], dtype=complex)

        assert np.allclose(self.program.state.interferometer, expected_interferometer)
