#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import pytest

from piquasso.core.registry import _register
from piquasso.api.instruction import Instruction
from piquasso.api.circuit import Circuit
from piquasso.api.state import State
from piquasso.api.program import Program


class TestProgramJSONParsing:
    @pytest.fixture
    def FakeInstruction(self):

        @_register
        class FakeInstruction(Instruction):
            def __init__(self, first_param, second_param):
                super().__init__(first_param=first_param, second_param=second_param)

        return FakeInstruction

    @pytest.fixture
    def FakeCircuit(self, FakeInstruction):

        @_register
        class FakeCircuit(Circuit):
            def get_instruction_map(self):
                return {
                    "FakeInstruction": FakeInstruction,
                }

        return FakeCircuit

    @pytest.fixture
    def FakeState(self, FakeCircuit):

        @_register
        class FakeState(State):
            circuit_class = FakeCircuit

            def __init__(self, foo, bar, d):
                self.foo = foo
                self.bar = bar
                self.d = d

        return FakeState

    @pytest.fixture
    def number_of_modes(self):
        return 420

    @pytest.fixture
    def state_mapping(self, number_of_modes):
        return {
            "type": "FakeState",
            "properties": {
                "foo": "fee",
                "bar": "beer",
                "d": number_of_modes,
            }
        }

    @pytest.fixture
    def instructions_mapping(self):
        return [
            {
                "type": "FakeInstruction",
                "properties": {
                    "params": {
                        "first_param": "first_param_value",
                        "second_param": "second_param_value",
                    },
                    "modes": ["some", "modes"],
                }
            },
            {
                "type": "FakeInstruction",
                "properties": {
                    "params": {
                        "first_param": "2nd_instructions_1st_param_value",
                        "second_param": "2nd_instructions_2nd_param_value",
                    },
                    "modes": ["some", "other", "modes"],
                }
            },
        ]

    def test_instantiation_using_mappings(
        self,
        FakeState,
        FakeCircuit,
        FakeInstruction,
        state_mapping,
        instructions_mapping,
        number_of_modes,
    ):
        program = Program.from_properties(
            {
                "state": state_mapping,
                "instructions": instructions_mapping,
            }
        )

        assert program.state.foo == "fee"
        assert program.state.bar == "beer"
        assert program.state.d == number_of_modes

        assert program.instructions[0].params == {
            "first_param": "first_param_value",
            "second_param": "second_param_value",
        }
        assert program.instructions[0].modes == ["some", "modes"]

        assert program.instructions[1].params == {
            "first_param": "2nd_instructions_1st_param_value",
            "second_param": "2nd_instructions_2nd_param_value",
        }
        assert program.instructions[1].modes == ["some", "other", "modes"]

    def test_from_json(
        self,
        FakeState,
        FakeCircuit,
        FakeInstruction,
        state_mapping,
        instructions_mapping,
        number_of_modes,
    ):
        json_ = json.dumps(
            {
                "state": state_mapping,
                "instructions": instructions_mapping,
            }
        )

        program = Program.from_json(json_)

        assert program.state.foo == "fee"
        assert program.state.bar == "beer"
        assert program.state.d == number_of_modes

        assert program.instructions[0].params == {
            "first_param": "first_param_value",
            "second_param": "second_param_value",
        }
        assert program.instructions[0].modes == ["some", "modes"]
        assert program.instructions[1].params == {
            "first_param": "2nd_instructions_1st_param_value",
            "second_param": "2nd_instructions_2nd_param_value",
        }
        assert program.instructions[1].modes == ["some", "other", "modes"]
