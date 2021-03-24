#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso import Q
from .test_program_base import TestProgramBase


class TestProgramExecution(TestProgramBase):
    def test_operation_execution(self, DummyOperation):
        operation = DummyOperation(param=420)
        with self.program:
            Q(0, 1) | operation

        self.program.execute()

        self.program.circuit.dummy_operation.assert_called_once_with(
            operation
        )

    def test_register_operation_from_left_hand_side(self, DummyOperation):
        operation = DummyOperation(param=420)
        with self.program:
            operation | Q(0, 1)

        self.program.execute()

        self.program.circuit.dummy_operation.assert_called_once_with(operation)
