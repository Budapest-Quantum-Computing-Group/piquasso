#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

from piquasso.sampling.backend import SamplingBackend
from piquasso.sampling.state import SamplingState


class TestSamplingBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        initial_state = SamplingState(1, 1, 1, 0, 0)
        self.backend = SamplingBackend(initial_state)

        permutation_matrix = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=complex)
        self.backend.state.interferometer = permutation_matrix

    def test_sampling_samples_number(self):
        shots = 100

        self.backend.sampling((shots,), None)

        assert len(self.backend.state.results) == shots,\
            f'Expected {shots} samples, ' \
            f'got: {len(self.backend.state.results)}'

    def test_sampling_mode_permutation(self):
        shots = 1

        self.backend.sampling((shots,), None)

        sample = self.backend.state.results[0]
        assert np.allclose(sample, [1, 0, 0, 1, 1]),\
            f'Expected [1, 0, 0, 1, 1], got: {sample}'

    def test_sampling_multiple_samples_for_permutation_interferometer(self):
        shots = 2

        self.backend.sampling((shots,), None)

        first_sample = self.backend.state.results[0]
        second_sample = self.backend.state.results[1]

        assert np.allclose(first_sample, second_sample),\
            f'Expected same samples, got: {first_sample} & {second_sample}'
