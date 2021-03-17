#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import piquasso as pq
from piquasso.api.errors import InvalidModes


def test_try_registering_nondistinct_modes():
    nondistinct_modes = (0, 0, 1)

    with pq.Program():
        with pytest.raises(InvalidModes):
            pq.Q(*nondistinct_modes) | pq.F()
