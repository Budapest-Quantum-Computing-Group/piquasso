#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.mode import Q
from piquasso.program import Program
from piquasso.tests.test_program_base import TestProgramBase


class TestProgramStacking(TestProgramBase):

    def test_single_operation_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(420)

        with self.program:
            Q(0, 1) | sub_program

        assert self.operations[0].modes == (0, 1)
        assert self.operations[0].params == (420,)

    def test_multiple_operation_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0) | DummyOperation(2) | DummyOperation(4)
            Q(2, 3) | DummyOperation(10)

        with self.program:
            Q(0, 1, 2, 3) | sub_program

        assert self.operations[0].modes == (0,)
        assert self.operations[0].params == (2,)

        assert self.operations[1].modes == (0,)
        assert self.operations[1].params == (4,)

        assert self.operations[2].modes == (2, 3)
        assert self.operations[2].params == (10,)

    def test_multiple_operation_mixed_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(10)

        with self.program:
            Q(2) | DummyOperation(2)
            Q(0, 1) | sub_program
            Q(3) | DummyOperation(0)

        assert self.operations[0].modes == (2,)
        assert self.operations[0].params == (2,)

        assert self.operations[1].modes == (0, 1)
        assert self.operations[1].params == (10,)

        assert self.operations[2].modes == (3,)
        assert self.operations[2].params == (0,)

    def test_multiple_modeless_operation_mixed_program_stacking(
            self,
            DummyOperation,
            DummyModelessOperation
    ):
        ModelessMode = None

        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(10)
            DummyModelessOperation(999)

        with self.program:
            Q(2) | DummyOperation(2)
            DummyModelessOperation(420)
            Q(0, 1) | sub_program
            Q(3) | DummyOperation(0)

        assert self.operations[0].modes == (2,)
        assert self.operations[0].params == (2,)

        assert self.operations[1].modes is ModelessMode
        assert self.operations[1].params == (420,)

        assert self.operations[2].modes == (0, 1)
        assert self.operations[2].params == (10,)

        assert self.operations[3].modes is ModelessMode
        assert self.operations[3].params == (999,)

        assert self.operations[4].modes == (3,)
        assert self.operations[4].params == (0,)

    def test_mixed_index_program_stacking(self, DummyOperation):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(10)
            Q(2, 3) | DummyOperation(100)

        with self.program:
            Q(0, 2, 1, 3) | sub_program

        assert self.operations[0].modes == (0, 2)
        assert self.operations[0].params == (10,)

        assert self.operations[1].modes == (1, 3)
        assert self.operations[1].params == (100,)

    def test_mixed_index_multiple_modeless_operation_program_stacking(
            self,
            DummyOperation,
            DummyModelessOperation
    ):
        ModelessMode = None

        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(1)
            Q(1, 2) | DummyOperation(2)
            DummyModelessOperation(3)

        with self.program:
            Q(0, 1, 2) | sub_program
            Q(1, 3, 2) | sub_program

        # Q(0, 1, 2) | sub_program
        assert self.operations[0].modes == (0, 1)
        assert self.operations[1].modes == (1, 2)
        assert self.operations[2].modes is ModelessMode

        assert self.operations[0].params == (1,)
        assert self.operations[1].params == (2,)
        assert self.operations[2].params == (3,)

        # Q(1, 3, 2) | sub_program
        assert self.operations[3].modes == (1, 3)
        assert self.operations[4].modes == (3, 2)
        assert self.operations[5].modes is ModelessMode

        assert self.operations[3].params == (1,)
        assert self.operations[4].params == (2,)
        assert self.operations[5].params == (3,)
