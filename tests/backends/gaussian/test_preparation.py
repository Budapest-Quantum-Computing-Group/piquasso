#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq

from piquasso.api.errors import StatePreparationError
from piquasso.api.constants import HBAR


def test_state_initialization_with_misshaped_m():
    valid_C = np.array(
        [
            [  1, 1j],
            [-1j,  1],
        ]
    )
    valid_G = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    misshaped_m = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=misshaped_m,
            G=valid_G,
            C=valid_C,
        )


def test_state_initialization_with_misshaped_G():
    valid_m = np.array([1, 2])
    valid_C = np.array(
        [
            [  1, 1j],
            [-1j,  1],
        ]
    )

    invalid_G = np.array(
        [
            [1, 2, 10000],
            [1, 1, 10000],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=valid_m,
            G=invalid_G,
            C=valid_C
        )


def test_state_initialization_with_non_symmetric_G():
    valid_m = np.array([1, 2])
    valid_C = np.array(
        [
            [  1, 1j],
            [-1j,  1],
        ]
    )

    non_symmetric_G = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=valid_m,
            G=non_symmetric_G,
            C=valid_C
        )


def test_state_initialization_with_misshaped_C():
    valid_m = np.array([1, 2])
    valid_G = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    misshaped_C = np.array(
        [
            [ 1, 3 + 2j, 10000],
            [1j,  1, 10000],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=valid_m,
            G=valid_G,
            C=misshaped_C,
        )


def test_state_initialization_with_non_selfadjoint_C():
    valid_m = np.array([1, 2])
    valid_G = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    non_selfadjoint_C = np.array(
        [
            [ 1, 3 + 2j],
            [1j,  1],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=valid_m,
            G=valid_G,
            C=non_selfadjoint_C,
        )


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
