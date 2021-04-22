#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pytest

import piquasso as pq


class TestProgramJSONParsing:
    @pytest.fixture
    def FakeInstruction(self):

        class FakeInstruction(pq.Instruction):
            def __init__(self, first_param, second_param):
                super().__init__(first_param=first_param, second_param=second_param)

        return FakeInstruction

    @pytest.fixture
    def FakeCircuit(self, FakeInstruction):

        class FakeCircuit(pq.Circuit):
            def get_instruction_map(self):
                return {
                    "FakeInstruction": FakeInstruction,
                }

        return FakeCircuit

    @pytest.fixture
    def FakeState(self, FakeCircuit):

        class FakeState(pq.State):
            circuit_class = FakeCircuit

            def __init__(self, foo, bar, d):
                self.foo = foo
                self.bar = bar
                self.d = d

        return FakeState

    @pytest.fixture(autouse=True)
    def setup(self, FakeState, FakeInstruction):
        class FakePlugin(pq.Plugin):
            classes = {
                "FakeState": FakeState,
                "FakeInstruction": FakeInstruction,
            }

        pq.use(FakePlugin)

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
        FakeInstruction,
        state_mapping,
        instructions_mapping,
        number_of_modes,
    ):
        program = pq.Program.from_properties(
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

        program = pq.Program.from_json(json_)

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
