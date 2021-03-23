#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso import Q
from .test_program_base import TestProgramBase


class TestProgramRegistry(TestProgramBase):

    def test_single_mode_single_operation_registry(self, DummyOperation):
        with self.program:
            Q(0) | DummyOperation(dummyparam=420)

        assert len(self.program.operations) == 1

        assert self.program.operations[0].modes == (0,)
        assert self.program.operations[0].params == {"dummyparam": 420}

    def test_single_mode_multiple_operation_registry(self, DummyOperation):
        with self.program:
            Q(0, 1) | DummyOperation(dummyparam=420) | DummyOperation(
                dummyparam1=42, dummyparam2=320
            )

        assert len(self.program.operations) == 2

        assert self.program.operations[0].modes == (0, 1)
        assert self.program.operations[0].params == {"dummyparam": 420}

        assert self.program.operations[1].modes == (0, 1)
        assert self.program.operations[1].params == {
            "dummyparam1": 42,
            "dummyparam2": 320,
        }

    def test_multiple_mode_single_operation_registry(self, DummyOperation):
        with self.program:
            Q(2, 1, 0) | DummyOperation(dummyparam1=421)
            Q(1) | DummyOperation(dummyparam2=1)
            Q(0, 2) | DummyOperation(dummyparam3=999)

        assert len(self.program.operations) == 3

        assert self.program.operations[0].modes == (2, 1, 0)
        assert self.program.operations[0].params == {"dummyparam1": 421}

        assert self.program.operations[1].modes == (1,)
        assert self.program.operations[1].params == {"dummyparam2": 1}

        assert self.program.operations[2].modes == (0, 2)
        assert self.program.operations[2].params == {"dummyparam3": 999}

    def test_multiple_mode_multiple_operation_registry(self, DummyOperation):
        with self.program:
            Q(4) | DummyOperation(param=2) | DummyOperation(param=0)
            Q(0, 2) | DummyOperation(param=999)
            Q(1, 0) | DummyOperation(param=1) | DummyOperation(param=9)

        assert len(self.program.operations) == 5

        assert self.program.operations[0].modes == (4,)
        assert self.program.operations[0].params == {"param": 2}

        assert self.program.operations[1].modes == (4,)
        assert self.program.operations[1].params == {"param": 0}

        assert self.program.operations[2].modes == (0, 2)
        assert self.program.operations[2].params == {"param": 999}

        assert self.program.operations[3].modes == (1, 0)
        assert self.program.operations[3].params == {"param": 1}

        assert self.program.operations[4].modes == (1, 0)
        assert self.program.operations[4].params == {"param": 9}
