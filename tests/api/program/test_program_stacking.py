#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

from piquasso import Q
from piquasso import Program
from .test_program_base import TestProgramBase


class TestProgramStacking(TestProgramBase):

    def test_single_instruction_program_stacking(self, DummyInstruction):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyInstruction(param=420)

        with self.program:
            Q(0, 1) | sub_program

        assert self.program.instructions[0].modes == (0, 1)
        assert self.program.instructions[0].params == {"param": 420}

    def test_multiple_instruction_program_stacking(self, DummyInstruction):
        sub_program = Program()
        with sub_program:
            Q(0) | DummyInstruction(param=2) | DummyInstruction(param=4)
            Q(2, 3) | DummyInstruction(param=10)

        with self.program:
            Q(0, 1, 2, 3) | sub_program

        assert self.program.instructions[0].modes == (0,)
        assert self.program.instructions[0].params == {"param": 2}

        assert self.program.instructions[1].modes == (0,)
        assert self.program.instructions[1].params == {"param": 4}

        assert self.program.instructions[2].modes == (2, 3)
        assert self.program.instructions[2].params == {"param": 10}

    def test_multiple_instruction_mixed_program_stacking(self, DummyInstruction):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyInstruction(param=10)

        with self.program:
            Q(2) | DummyInstruction(param=2)
            Q(0, 1) | sub_program
            Q(3) | DummyInstruction(param=0)

        assert self.program.instructions[0].modes == (2,)
        assert self.program.instructions[0].params == {"param": 2}

        assert self.program.instructions[1].modes == (0, 1)
        assert self.program.instructions[1].params == {"param": 10}

        assert self.program.instructions[2].modes == (3,)
        assert self.program.instructions[2].params == {"param": 0}

    def test_mixed_index_program_stacking(self, DummyInstruction):
        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyInstruction(param=10)
            Q(2, 3) | DummyInstruction(param=100)

        with self.program:
            Q(0, 2, 1, 3) | sub_program

        assert self.program.instructions[0].modes == (0, 2)
        assert self.program.instructions[0].params == {"param": 10}

        assert self.program.instructions[1].modes == (1, 3)
        assert self.program.instructions[1].params == {"param": 100}

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

    def test_main_program_inherits_state_and_instructions(
        self, FakeState, DummyInstruction
    ):
        with Program() as preparation:
            Q() | FakeState()

            Q(0, 1) | DummyInstruction(param=10)
            Q(2, 3) | DummyInstruction(param=20)

        with Program() as main:
            Q(0, 1, 2, 3) | preparation

        assert main.state is not None
        assert main.circuit.state is main.state
        assert main.state is not preparation.state, (
            "The state should be copied from the subprogram due to the state's "
            "mutability."
        )

        assert main.instructions[0] == DummyInstruction(param=10).on_modes(0, 1)
        assert main.instructions[1] == DummyInstruction(param=20).on_modes(2, 3)

    def test_main_program_inherits_state_and_instructions_without_modes_specified(
        self, FakeState, DummyInstruction
    ):
        state = FakeState()
        with Program() as preparation:
            Q() | state

            Q(0, 1) | DummyInstruction(param=10)
            Q(2, 3) | DummyInstruction(param=20)

        with Program() as main:
            first_instruction = Q() | preparation

        assert first_instruction.modes == tuple(range(state.d))

        assert main.state is not None
        assert main.circuit.state is main.state
        assert main.state is not preparation.state, (
            "The state should be copied from the subprogram due to the state's "
            "mutability."
        )

        assert main.instructions[0] == DummyInstruction(param=10).on_modes(0, 1)
        assert main.instructions[1] == DummyInstruction(param=20).on_modes(2, 3)

    def test_state_collision_raises_RuntimeError(self, FakeState):
        with Program() as preparation:
            Q() | FakeState()

        with pytest.raises(RuntimeError) as error:
            with self.program:
                Q() | preparation

        assert error.value.args[0] == (
            "The current program already has a state registered of type '_FakeState'."
        )
