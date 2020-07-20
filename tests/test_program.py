#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso.program import Program
from piquasso.context import Context
from piquasso.gates import B
from piquasso.mode import Q
from piquasso.state import State
from piquasso.backend import FockBackend


@pytest.fixture
def dummy_state():
    return State.from_state_vector([1, 0])


@pytest.fixture(scope="session")
def tolerance():
    return 10E-9


def test_current_program_in_program_context(dummy_state):
    program = Program(state=dummy_state, backend=FockBackend)

    with program:
        assert Context.current_program is program

    assert Context.current_program is None


def test_program(dummy_state, tolerance):
    program = Program(state=dummy_state, backend=FockBackend)

    with program:
        Q(0, 1) | B(0.1, 0.4) | B(0.5, 0.3)

    program.execute()

    assert len(program.instructions) == 2
    assert np.abs(program.state.trace() - 1) < tolerance
