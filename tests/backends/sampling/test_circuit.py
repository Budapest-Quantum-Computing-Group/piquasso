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


class TestSampling:
    @pytest.fixture(autouse=True)
    def setup(self):
        initial_state = pq.SamplingState(1, 1, 1, 0, 0)
        self.program = pq.Program(initial_state)

    def test_program(self):
        U = np.array([
            [.5, 0, 0],
            [0, .5j, 0],
            [0, 0, -1]
        ], dtype=complex)

        with self.program:
            pq.Q(0, 1) | pq.Beamsplitter(.5)
            pq.Q(1, 2, 3) | pq.Interferometer(U)
            pq.Q(3) | pq.Phaseshifter(.5)
            pq.Q(4) | pq.Phaseshifter(.5)
            pq.Q() | pq.Sampling(shots=10)

        results = self.program.execute()

        assert len(results) == 1
        assert len(results[0].samples) == 10

    def test_interferometer(self):
        U = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=complex)
        with self.program:
            pq.Q(4, 3, 1) | pq.Interferometer(U)
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
            pq.Q(2) | pq.Phaseshifter(phi)
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
            pq.Q(1, 3) | pq.Beamsplitter(theta, phi)

        self.program.execute()

        t = np.cos(theta)
        r = np.exp(1j * phi) * np.sin(theta)
        rc = np.conj(r)
        expected_interferometer = np.array(
            [
                [1, 0, 0,   0, 0],
                [0, t, 0, -rc, 0],
                [0, 0, 1,   0, 0],
                [0, r, 0,   t, 0],
                [0, 0, 0,   0, 1],
            ],
            dtype=complex
        )

        assert np.allclose(self.program.state.interferometer, expected_interferometer)

    def test_lossy_program(self):
        r'''
            This test checks the average number of particles in the lossy BS.
            We expect average number to be smaller than initial one.
        '''
        losses = 0.5
        U = np.eye(5) * losses
        U[0][0] = 0  # Ensure that at least one particle is lost.
        self.program.state.is_lossy = True

        with self.program:
            pq.Q(0, 1, 2, 3, 4) | pq.Interferometer(U)
            pq.Q() | pq.Sampling(shots=1)

        results = self.program.execute()
        sample = results[0].samples[0]
        assert sum(sample) < sum(self.program.state.initial_state)
