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

import pytest
import numpy as np

import piquasso as pq

from piquasso.api.errors import InvalidState


def test_state_initialization_with_misshaped_mean():
    misshaped_mean = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Mean(misshaped_mean)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_misshaped_covariance():
    misshaped_cov = np.array(
        [
            [1, 2, 10000],
            [1, 1, 10000],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(misshaped_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_nonsymmetric_covariance():
    nonsymmetric_cov = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(nonsymmetric_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_nonpositive_covariance():
    nonpositive_cov = np.array(
        [
            [1, 0],
            [0, -1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(nonpositive_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_vacuum_resets_the_state(state):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=state.d)

    new_state = simulator.execute(program, initial_state=state).state

    assert np.allclose(
        new_state.xpxp_mean_vector,
        np.zeros(2 * new_state.d),
    )
    assert np.allclose(
        new_state.xpxp_covariance_matrix,
        np.identity(2 * new_state.d) * simulator.config.hbar,
    )
