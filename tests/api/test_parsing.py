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

import pytest

import piquasso as pq


@pytest.fixture(scope="module")
def FakeInstruction():
    class FakeInstruction(pq.Instruction):
        def __init__(self, first_param, second_param):
            super().__init__(
                params=dict(
                    first_param=first_param,
                    second_param=second_param,
                ),
            )

    return FakeInstruction


@pytest.fixture(scope="module")
def FakeState():
    class FakeState(pq.State):
        _instruction_map = {
            "FakeInstruction": "_fake_instruction",
        }

        def __init__(self, foo, bar, d):
            super().__init__()
            self.foo = foo
            self.bar = bar
            self._d = d

        @property
        def d(self) -> int:
            return self._d

        def _fake_instruction(self, instruction, state):
            pass

        def get_particle_detection_probability(self, occupation_number: tuple) -> float:
            raise NotImplementedError

    return FakeState


@pytest.fixture(autouse=True, scope="module")
def setup(FakeState, FakeInstruction):
    class FakePlugin(pq.Plugin):
        classes = {
            "FakeState": FakeState,
            "FakeInstruction": FakeInstruction,
        }

    pq.registry.use_plugin(FakePlugin)


@pytest.fixture
def number_of_modes():
    return 420


@pytest.fixture
def state_dict(number_of_modes):
    return {
        "type": "FakeState",
        "attributes": {
            "constructor_kwargs": {
                "foo": "fee",
                "bar": "beer",
                "d": number_of_modes,
            }
        },
    }


@pytest.fixture
def instructions_dict():
    return {
        "instructions": [
            {
                "type": "FakeInstruction",
                "attributes": {
                    "constructor_kwargs": {
                        "first_param": "first_param_value",
                        "second_param": "second_param_value",
                    },
                    "modes": ["some", "modes"],
                },
            },
            {
                "type": "FakeInstruction",
                "attributes": {
                    "constructor_kwargs": {
                        "first_param": "2nd_instructions_1st_param_value",
                        "second_param": "2nd_instructions_2nd_param_value",
                    },
                    "modes": ["some", "other", "modes"],
                },
            },
        ]
    }


def test_state_instantiation_using_dicts(
    state_dict,
    number_of_modes,
):
    state = pq.State.from_dict(state_dict)

    assert state.foo == "fee"
    assert state.bar == "beer"
    assert state.d == number_of_modes


def test_program_instantiation_using_dicts(
    instructions_dict,
):
    program = pq.Program.from_dict(instructions_dict)

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
