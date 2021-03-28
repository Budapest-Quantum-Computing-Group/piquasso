#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import piquasso as pq


def test_instruction_execution(program):
    instruction = pq.DummyInstruction(param=420)
    with program:
        pq.Q(0, 1) | instruction

    program.execute()

    program.circuit.dummy_instruction.assert_called_once_with(instruction)


def test_register_instruction_from_left_hand_side(program):
    instruction = pq.DummyInstruction(param=420)
    with program:
        instruction | pq.Q(0, 1)

    program.execute()

    program.circuit.dummy_instruction.assert_called_once_with(instruction)
