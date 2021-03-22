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
