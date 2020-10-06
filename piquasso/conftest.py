#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest


@pytest.fixture(scope="session")
def tolerance():
    return 1E-9
