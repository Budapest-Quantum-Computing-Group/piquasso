#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq

from piquasso.api.errors import StatePreparationError


def test_state_initialization_with_non_symmetric_G():
    valid_m = np.array([1, 2])
    valid_C = np.array(
        [
            [  1, 1j],
            [-1j,  1],
        ]
    )

    invalid_G = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(m=valid_m, G=invalid_G, C=valid_C)


def test_state_initialization_with_non_selfadjoint_C():
    valid_m = np.array([1, 2])
    valid_G = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    invalid_C = np.array(
        [
            [ 1, 3 + 2j],
            [1j,  1],
        ]
    )

    with pytest.raises(StatePreparationError):
        pq.GaussianState(
            m=valid_m,
            G=valid_G,
            C=invalid_C
        )
