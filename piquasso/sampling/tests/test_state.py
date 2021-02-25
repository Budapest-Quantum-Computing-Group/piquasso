#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

from ..state import SamplingState


class TestSamplingState:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.state = SamplingState(1, 1, 1, 0, 0)

    def test_initial_state(self):
        expected_initial_state = [1, 1, 1, 0, 0]
        assert np.allclose(self.state.initial_state, expected_initial_state)

    def test_results_init(self):
        assert self.state.results is None

    def test_interferometer_init(self):
        expected_interferometer = np.diag(np.ones(self.state.d, dtype=complex))
        assert np.allclose(self.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_neighbouring_modes(self):
        U = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=complex)
        self.state.multiply_interferometer_on_modes(U, [0, 1, 2])

        expected_interferometer = np.array([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ], dtype=complex)

        assert np.allclose(self.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_gaped_modes(self):
        U = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=complex)
        self.state.multiply_interferometer_on_modes(U, [0, 1, 4])

        expected_interferometer = np.array([
            [1, 2, 0, 0, 3],
            [4, 5, 0, 0, 6],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [7, 8, 0, 0, 9],
        ], dtype=complex)

        assert np.allclose(self.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_reversed_gaped_modes(self):
        U = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=complex)
        self.state.multiply_interferometer_on_modes(U, [4, 3, 1])

        expected_interferometer = np.array([
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ], dtype=complex)

        assert np.allclose(self.state.interferometer, expected_interferometer)
