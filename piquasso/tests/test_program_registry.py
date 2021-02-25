#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.mode import Q
from piquasso.tests.test_program_base import TestProgramBase


class TestProgramRegistry(TestProgramBase):

    def test_single_mode_single_operation_registry(self, DummyOperation):
        with self.program:
            Q(0) | DummyOperation(420)

        assert len(self.program.operations) == 1

        assert self.program.operations[0].modes == (0,)
        assert self.program.operations[0].params == (420,)

    def test_single_mode_multiple_operation_registry(self, DummyOperation):
        with self.program:
            Q(0, 1) | DummyOperation(420) | DummyOperation(42, 320)

        assert len(self.program.operations) == 2

        assert self.program.operations[0].modes == (0, 1)
        assert self.program.operations[0].params == (420,)

        assert self.program.operations[1].modes == (0, 1)
        assert self.program.operations[1].params == (42, 320)

    def test_multiple_mode_single_operation_registry(self, DummyOperation):
        with self.program:
            Q(2, 1, 0) | DummyOperation(421)
            Q(1) | DummyOperation(1)
            Q(0, 2) | DummyOperation(999)

        assert len(self.program.operations) == 3

        assert self.program.operations[0].modes == (2, 1, 0)
        assert self.program.operations[0].params == (421,)

        assert self.program.operations[1].modes == (1,)
        assert self.program.operations[1].params == (1,)

        assert self.program.operations[2].modes == (0, 2)
        assert self.program.operations[2].params == (999,)

    def test_multiple_mode_multiple_operation_registry(self, DummyOperation):
        with self.program:
            Q(4) | DummyOperation(2) | DummyOperation(0)
            Q(0, 2) | DummyOperation(999)
            Q(1, 0) | DummyOperation(1) | DummyOperation(9)

        assert len(self.program.operations) == 5

        assert self.program.operations[0].modes == (4,)
        assert self.program.operations[0].params == (2,)

        assert self.program.operations[1].modes == (4,)
        assert self.program.operations[1].params == (0,)

        assert self.program.operations[2].modes == (0, 2)
        assert self.program.operations[2].params == (999,)

        assert self.program.operations[3].modes == (1, 0)
        assert self.program.operations[3].params == (1,)

        assert self.program.operations[4].modes == (1, 0)
        assert self.program.operations[4].params == (9,)


class TestModelessOperationRegistry(TestProgramBase):

    def test_single_modeless_operation_registry(self, DummyModelessOperation):
        ModelessMode = None

        with self.program:
            DummyModelessOperation(42)

        assert len(self.program.operations) == 1
        assert self.program.operations[0].params == (42,)
        assert self.program.operations[0].modes is ModelessMode

    def test_multiple_modeless_operation_registry(self, DummyModelessOperation):
        with self.program:
            DummyModelessOperation(4)
            DummyModelessOperation(2)
            DummyModelessOperation(0)

        assert len(self.program.operations) == 3
        assert self.program.operations[0].params == (4,)
        assert self.program.operations[1].params == (2,)
        assert self.program.operations[2].params == (0,)

    def test_mixed_operation_registry(self, DummyOperation, DummyModelessOperation):
        ModelessMode = None

        with self.program:
            Q(0, 1) | DummyOperation(0)
            DummyModelessOperation(10)
            Q(3) | DummyOperation(2) | DummyOperation(1)
            DummyModelessOperation(9)

        assert len(self.program.operations) == 5

        assert self.program.operations[0].params == (0,)
        assert self.program.operations[0].modes == (0, 1)

        assert self.program.operations[1].params == (10,)
        assert self.program.operations[1].modes is ModelessMode

        assert self.program.operations[2].params == (2,)
        assert self.program.operations[2].modes == (3,)

        assert self.program.operations[3].params == (1,)
        assert self.program.operations[3].modes == (3,)

        assert self.program.operations[4].params == (9,)
        assert self.program.operations[4].modes is ModelessMode
