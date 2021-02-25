#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso import Q
from piquasso.tests.test_program_base import TestProgramBase


class TestProgramExecution(TestProgramBase):
    def test_operation_execution(self, DummyOperation):
        operation = DummyOperation(420)
        with self.program:
            Q(0, 1) | operation

        self.program.execute()

        self.program.backend.dummy_operation.assert_called_once_with(
            self.program.backend,
            operation
        )

    def test_modeless_operation_execution(self, DummyModelessOperation):
        with self.program:
            operation = DummyModelessOperation(321)

        self.program.execute()

        self.program.backend.dummy_modeless_operation.assert_called_once_with(
            self.program.backend,
            operation
        )

    def test_mixed_operation_execution(self, DummyOperation, DummyModelessOperation):
        with self.program:
            Q(0) | DummyOperation(420)
            DummyModelessOperation(321)

        self.program.execute()

        self.program.backend.dummy_operation.assert_called_once()
        self.program.backend.dummy_modeless_operation.assert_called_once()
