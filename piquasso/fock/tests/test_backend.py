#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

from ..backend import FockBackend
from ..state import FockState


@pytest.fixture(scope="session")
def tolerance():
    return 10E-10


class TestFockBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        state = FockState.from_state_vector([1, 0])
        self.backend = FockBackend(state)

    def test_phaseshift(self, tolerance):
        modes = (0, 1)

        self.backend.beamsplitter((0.1, 0.4), modes)
        self.backend.beamsplitter((0.5, 0.3), modes)

        assert np.abs(self.backend.state.trace() - 1) < tolerance
