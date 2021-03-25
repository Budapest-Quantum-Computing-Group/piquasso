#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso import Q
from .test_program_base import TestProgramBase


class TestProgramExecution(TestProgramBase):
    def test_instruction_execution(self, DummyInstruction):
        instruction = DummyInstruction(param=420)
        with self.program:
            Q(0, 1) | instruction

        self.program.execute()

        self.program.circuit.dummy_instruction.assert_called_once_with(
            instruction
        )

    def test_register_instruction_from_left_hand_side(self, DummyInstruction):
        instruction = DummyInstruction(param=420)
        with self.program:
            instruction | Q(0, 1)

        self.program.execute()

        self.program.circuit.dummy_instruction.assert_called_once_with(instruction)
