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


class TestSamplingState:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.state = pq.SamplingState(1, 1, 1, 0, 0)

    def test_initial_state(self):
        expected_initial_state = [1, 1, 1, 0, 0]
        assert np.allclose(self.state.initial_state, expected_initial_state)

    def test_results_init(self):
        assert self.state.results is None

    def test_interferometer_init(self):
        expected_interferometer = np.diag(np.ones(self.state.d, dtype=complex))
        assert np.allclose(self.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_neighbouring_modes(self):
        U = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=complex,
        )

        with pq.Program() as program:
            pq.Q() | self.state

            pq.Q(0, 1, 2) | pq.Interferometer(U)

        program.execute()

        expected_interferometer = np.array(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0],
                [7, 8, 9, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=complex,
        )

        assert np.allclose(program.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_gaped_modes(self):
        U = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=complex,
        )

        with pq.Program() as program:
            pq.Q() | self.state

            pq.Q(0, 1, 4) | pq.Interferometer(U)

        program.execute()

        expected_interferometer = np.array(
            [
                [1, 2, 0, 0, 3],
                [4, 5, 0, 0, 6],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [7, 8, 0, 0, 9],
            ],
            dtype=complex,
        )

        assert np.allclose(program.state.interferometer, expected_interferometer)

    def test_multiple_interferometer_on_reversed_gaped_modes(self):
        U = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            dtype=complex,
        )

        with pq.Program() as program:
            pq.Q() | self.state

            pq.Q(4, 3, 1) | pq.Interferometer(U)

        program.execute()

        expected_interferometer = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 9, 0, 8, 7],
                [0, 0, 1, 0, 0],
                [0, 6, 0, 5, 4],
                [0, 3, 0, 2, 1],
            ],
            dtype=complex,
        )

        assert np.allclose(program.state.interferometer, expected_interferometer)

    def test_distribution(self):
        input_state = pq.SamplingState(1, 1, 0)

        U = np.asarray([[1, 0, 0],
                        [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
                        [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j]])

        input_state.interferometer = U

        probabilities = input_state.get_fock_probabilities()
        expected_result = [0.0, 0.30545762086020883, 0.6945423895038292, 0.0, 0.0, 0.0]

        assert np.allclose(probabilities, expected_result)
