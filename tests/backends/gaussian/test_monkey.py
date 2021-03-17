#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import random
import numpy as np

from scipy.stats import unitary_group

import piquasso as pq


@pytest.mark.monkey
@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_random_gaussianstate(d):
    with pq.Program() as prepare_random_state:
        pq.Q() | pq.GaussianState.create_vacuum(d=d)

        for i in range(d):
            r = np.random.normal()
            theta = random.uniform(0, 2 * np.pi)

            pq.Q(i) | pq.S(amp=r, theta=theta)

        random_unitary = np.array(unitary_group.rvs(d))

        pq.Q(*range(d)) | pq.Interferometer(random_unitary)

    with pq.Program() as program:
        pq.Q(*range(d)) | prepare_random_state

    program.state.validate()
