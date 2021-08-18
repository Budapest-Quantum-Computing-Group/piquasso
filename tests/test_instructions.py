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

from piquasso.api.instruction import Instruction
from piquasso.api.errors import InvalidParameter


def test_instruction_initialization_from_dict():
    instruction_dict = {
        "type": "DummyInstruction",
        "attributes": {
            "constructor_kwargs": {
                "first_param": "first_param_value",
                "second_param": "second_param_value"
            },
            "modes": ["some", "modes"],
        }
    }

    class DummyInstruction(Instruction):
        def __init__(self, first_param, second_param):
            super().__init__(
                params=dict(
                    first_param=first_param,
                    second_param=second_param,
                ),
            )

    class DummyPlugin:
        classes = {
            "DummyInstruction": DummyInstruction,
        }

    pq.use(DummyPlugin)

    instruction = Instruction.from_dict(instruction_dict)

    assert isinstance(instruction, DummyInstruction)
    assert instruction.params == {
        "first_param": "first_param_value",
        "second_param": "second_param_value"
    }
    assert instruction.modes == ["some", "modes"]


def test_displacement_raises_InvalidParameter_for_redundant_parameters():
    with pytest.raises(InvalidParameter):
        pq.Displacement(alpha=1, r=2, phi=3)
