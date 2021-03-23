#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

from piquasso import Q
from piquasso import Program
from .test_program_base import TestProgramBase


class TestProgramStacking(TestProgramBase):

    def test_single_operation_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(param=420)

        with self.program:
            Q(0, 1) | sub_program

        assert self.program.operations[0].modes == (0, 1)
        assert self.program.operations[0].params == {"param": 420}

    def test_multiple_operation_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0) | DummyOperation(param=2) | DummyOperation(param=4)
            Q(2, 3) | DummyOperation(param=10)

        with self.program:
            Q(0, 1, 2, 3) | sub_program

        assert self.program.operations[0].modes == (0,)
        assert self.program.operations[0].params == {"param": 2}

        assert self.program.operations[1].modes == (0,)
        assert self.program.operations[1].params == {"param": 4}

        assert self.program.operations[2].modes == (2, 3)
        assert self.program.operations[2].params == {"param": 10}

    def test_multiple_operation_mixed_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(param=10)

        with self.program:
            Q(2) | DummyOperation(param=2)
            Q(0, 1) | sub_program
            Q(3) | DummyOperation(param=0)

        assert self.program.operations[0].modes == (2,)
        assert self.program.operations[0].params == {"param": 2}

        assert self.program.operations[1].modes == (0, 1)
        assert self.program.operations[1].params == {"param": 10}

        assert self.program.operations[2].modes == (3,)
        assert self.program.operations[2].params == {"param": 0}

    def test_mixed_index_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(param=10)
            Q(2, 3) | DummyOperation(param=100)

        with self.program:
            Q(0, 2, 1, 3) | sub_program

        assert self.program.operations[0].modes == (0, 2)
        assert self.program.operations[0].params == {"param": 10}

        assert self.program.operations[1].modes == (1, 3)
        assert self.program.operations[1].params == {"param": 100}

    def test_main_program_inherits_state(self, FakeState):
        with Program() as preparation:
            Q() | FakeState()

        with Program() as main:
            Q() | preparation

        assert main.state is not None
        assert main.circuit.state is main.state
        assert main.state is not preparation.state, (
            "The state should be copied from the subprogram due to the state's "
            "mutability."
        )

    def test_state_collision_raises_RuntimeError(self, FakeState):
        with Program() as preparation:
            Q() | FakeState()

        with pytest.raises(RuntimeError) as error:
            with self.program:
                Q() | preparation

        assert error.value.args[0] == (
            "The current program already has a state registered of type '_FakeState'."
        )
