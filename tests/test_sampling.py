#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

import piquasso as pq


class TestSampling:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.program = pq.Program(
            state=pq.SamplingState(1, 1, 1, 0, 0),
            backend_class=pq.SamplingBackend,
        )

        permutation_matrix = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=complex)

        self.program.state.interferometer = permutation_matrix

    def test_sampling_samples_number(self):
        shots = 100

        with self.program:
            pq.Sampling(shots)

        self.program.execute()

        assert len(self.program.state.results) == shots,\
            f'Expected {shots} samples, ' \
            f'got: {len(self.program.state.results)}'

    def test_sampling_mode_permutation(self):
        shots = 1

        with self.program:
            pq.Sampling(shots)

        self.program.execute()

        sample = self.program.state.results[0]
        assert np.allclose(sample, [1, 0, 0, 1, 1]),\
            f'Expected [1, 0, 0, 1, 1], got: {sample}'

    def test_sampling_multiple_samples_for_permutation_interferometer(self):
        shots = 2

        with self.program:
            pq.Sampling(shots)

        self.program.execute()

        first_sample = self.program.state.results[0]
        second_sample = self.program.state.results[1]

        assert np.allclose(first_sample, second_sample),\
            f'Expected same samples, got: {first_sample} & {second_sample}'
