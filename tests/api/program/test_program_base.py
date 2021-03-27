#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.api.circuit import Circuit
from piquasso.api.state import State
from piquasso.api.program import Program
from piquasso.api.instruction import Instruction


class TestProgramBase:
    @pytest.fixture
    def DummyInstruction(self):
        class _DummyInstruction(Instruction):
            pass

        return _DummyInstruction

    @pytest.fixture
    def FakeCircuit(self, DummyInstruction):
        class _FakeCircuit(Circuit):
            dummy_instruction = Mock(name="dummy_instruction")

            def get_instruction_map(self):
                return {
                    DummyInstruction.__name__: self.dummy_instruction,
                }

        return _FakeCircuit

    @pytest.fixture
    def FakeState(self, FakeCircuit):
        class _FakeState(State):
            circuit_class = FakeCircuit
            d = 42

        return _FakeState

    @pytest.fixture(autouse=True)
    def setup(self, FakeState):
        self.state = FakeState()

        self.program = Program(state=self.state)


def test_program_copy():
    program = Program()

    program_copy = program.copy()

    assert program_copy is not program
