#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso import Q
from .test_program_base import TestProgramBase


class TestProgramRegistry(TestProgramBase):

    def test_single_mode_single_instruction_registry(self, DummyInstruction):
        with self.program:
            Q(0) | DummyInstruction(dummyparam=420)

        assert len(self.program.instructions) == 1

        assert self.program.instructions[0].modes == (0,)
        assert self.program.instructions[0].params == {"dummyparam": 420}

    def test_single_mode_multiple_instruction_registry(self, DummyInstruction):
        with self.program:
            Q(0, 1) | DummyInstruction(dummyparam=420) | DummyInstruction(
                dummyparam1=42, dummyparam2=320
            )

        assert len(self.program.instructions) == 2

        assert self.program.instructions[0].modes == (0, 1)
        assert self.program.instructions[0].params == {"dummyparam": 420}

        assert self.program.instructions[1].modes == (0, 1)
        assert self.program.instructions[1].params == {
            "dummyparam1": 42,
            "dummyparam2": 320,
        }

    def test_multiple_mode_single_instruction_registry(self, DummyInstruction):
        with self.program:
            Q(2, 1, 0) | DummyInstruction(dummyparam1=421)
            Q(1) | DummyInstruction(dummyparam2=1)
            Q(0, 2) | DummyInstruction(dummyparam3=999)

        assert len(self.program.instructions) == 3

        assert self.program.instructions[0].modes == (2, 1, 0)
        assert self.program.instructions[0].params == {"dummyparam1": 421}

        assert self.program.instructions[1].modes == (1,)
        assert self.program.instructions[1].params == {"dummyparam2": 1}

        assert self.program.instructions[2].modes == (0, 2)
        assert self.program.instructions[2].params == {"dummyparam3": 999}

    def test_multiple_mode_multiple_instruction_registry(self, DummyInstruction):
        with self.program:
            Q(4) | DummyInstruction(param=2) | DummyInstruction(param=0)
            Q(0, 2) | DummyInstruction(param=999)
            Q(1, 0) | DummyInstruction(param=1) | DummyInstruction(param=9)

        assert len(self.program.instructions) == 5

        assert self.program.instructions[0].modes == (4,)
        assert self.program.instructions[0].params == {"param": 2}

        assert self.program.instructions[1].modes == (4,)
        assert self.program.instructions[1].params == {"param": 0}

        assert self.program.instructions[2].modes == (0, 2)
        assert self.program.instructions[2].params == {"param": 999}

        assert self.program.instructions[3].modes == (1, 0)
        assert self.program.instructions[3].params == {"param": 1}

        assert self.program.instructions[4].modes == (1, 0)
        assert self.program.instructions[4].params == {"param": 9}

    def test_instruction_registration_with_no_modes_is_resolved_to_all_modes(
        self,
        DummyInstruction,
    ):
        with self.program:
            Q() | DummyInstruction(param="some-parameter")

        self.program.execute()

        assert self.program.instructions[0].modes == tuple(range(self.program.state.d))

    def test_instruction_registration_with_all_keyword_is_resolved_to_all_modes(
        self,
        DummyInstruction,
    ):
        with self.program:
            Q(all) | DummyInstruction(param="some-parameter")

        self.program.execute()

        assert self.program.instructions[0].modes == tuple(range(self.program.state.d))
