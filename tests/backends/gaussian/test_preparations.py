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
from piquasso.api.constants import HBAR


def test_state_initialization_with_misshaped_mean():
    misshaped_mean = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=1)

        pq.Q() | pq.Mean(misshaped_mean)

    with pytest.raises(InvalidState):
        program.execute()


def test_state_initialization_with_misshaped_cov():
    misshaped_cov = np.array(
        [
            [1, 2, 10000],
            [1, 1, 10000],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=1)

        pq.Q() | pq.Covariance(misshaped_cov)

    with pytest.raises(InvalidState):
        program.execute()


def test_state_initialization_with_nonsymmetric_cov():
    nonsymmetric_cov = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=1)

        pq.Q() | pq.Covariance(nonsymmetric_cov)

    with pytest.raises(InvalidState):
        program.execute()


def test_state_initialization_with_nonpositive_cov():
    nonpositive_cov = np.array(
        [
            [1,  0],
            [0, -1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=1)

        pq.Q() | pq.Covariance(nonpositive_cov)

    with pytest.raises(InvalidState):
        program.execute()


def test_vacuum_resets_the_state(program):
    with program:
        pq.Q() | pq.Vacuum()

    program.execute()

    state = program.state

    assert np.allclose(
        program.state.mean,
        np.zeros(2 * state.d),
    )
    assert np.allclose(
        program.state.cov,
        np.identity(2 * state.d) * HBAR,
    )
