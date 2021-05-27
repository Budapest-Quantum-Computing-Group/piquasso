#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import piquasso as pq
from piquasso.api.errors import InvalidParameter


class TestSampling:
    @pytest.fixture(autouse=True)
    def setup(self):
        permutation_matrix = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=complex
        )

        with pq.Program() as program:
            pq.Q() | pq.SamplingState(5) | pq.OccupationNumbers((1, 1, 1, 0, 0))

            pq.Q() | pq.Interferometer(permutation_matrix)

        self.program = program

    def test_sampling_raises_InvalidParameter_for_negative_shot_value(self):
        invalid_shots = -1

        with self.program:
            with pytest.raises(InvalidParameter):
                pq.Q() | pq.Sampling(invalid_shots)

    def test_sampling_samples_number(self):
        shots = 100

        with self.program:
            pq.Q() | pq.Sampling(shots)

        self.program.execute()

        assert len(self.program.results[-1].samples) == shots,\
            f'Expected {shots} samples, ' \
            f'got: {len(self.program.state.results)}'

    def test_sampling_mode_permutation(self):
        shots = 1

        with self.program:
            pq.Q() | pq.Sampling(shots)

        self.program.execute()

        sample = self.program.results[-1].samples[0]
        assert np.allclose(sample, [1, 0, 0, 1, 1]),\
            f'Expected [1, 0, 0, 1, 1], got: {sample}'

    def test_sampling_multiple_samples_for_permutation_interferometer(self):
        shots = 2

        with self.program:
            pq.Q() | pq.Sampling(shots)

        self.program.execute()

        samples = self.program.results[-1].samples
        first_sample = samples[0]
        second_sample = samples[1]

        assert np.allclose(first_sample, second_sample),\
            f'Expected same samples, got: {first_sample} & {second_sample}'

    def test_mach_zehnder(self):
        int_ = np.pi/3
        ext = np.pi/4

        with self.program:
            pq.Q(0, 1) | pq.MachZehnder(int_=int_, ext=ext)
            pq.Q() | pq.Sampling(shots=1)

        self.program.execute()

    def test_fourier(self):
        with self.program:
            pq.Q(0) | pq.Fourier()
            pq.Q() | pq.Sampling(shots=1)

        self.program.execute()

    def test_loss(self):
        with self.program:
            pq.Q(all) | pq.Loss(transmissivity=0.9)

            pq.Q() | pq.Sampling(shots=1)

        self.program.execute()

        assert self.program.state.is_lossy
