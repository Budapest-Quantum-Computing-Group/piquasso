#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np

import piquasso as pq


@pytest.fixture
def program():
    with pq.Program() as initialization:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | pq.D(alpha=1)
        pq.Q(1) | pq.D(alpha=1j)
        pq.Q(2) | pq.D(alpha=np.exp(1j * np.pi/4))

        pq.Q(0) | pq.S(np.log(2), phi=np.pi / 2)
        pq.Q(1) | pq.S(np.log(1), phi=np.pi / 4)
        pq.Q(2) | pq.S(np.log(2), phi=np.pi / 2)

    initialization.execute()
    initialization.state.validate()

    return pq.Program(state=initialization.state)
