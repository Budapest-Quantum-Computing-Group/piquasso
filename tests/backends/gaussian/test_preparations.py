#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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

    state = pq.GaussianState(d=1)

    with pytest.raises(InvalidState):
        state.mean = misshaped_mean


def test_state_initialization_with_misshaped_cov():
    misshaped_cov = np.array(
        [
            [1, 2, 10000],
            [1, 1, 10000],
        ]
    )

    state = pq.GaussianState(d=1)

    with pytest.raises(InvalidState):
        state.cov = misshaped_cov


def test_state_initialization_with_non_symmetric_cov():
    non_symmetric_cov = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    state = pq.GaussianState(d=1)

    with pytest.raises(InvalidState):
        state.cov = non_symmetric_cov


def test_state_initialization_with_nonpositive_cov():
    nonpositive_cov = np.array(
        [
            [1,  0],
            [0, -1],
        ]
    )

    state = pq.GaussianState(d=1)

    with pytest.raises(InvalidState):
        state.cov = nonpositive_cov


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
