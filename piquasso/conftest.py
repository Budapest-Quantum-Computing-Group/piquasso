#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from scipy.stats import unitary_group


@pytest.fixture
def dummy_unitary():
    def func(d):
        return np.array(unitary_group.rvs(d))

    return func


@pytest.fixture(scope="session")
def tolerance():
    return 1E-9
