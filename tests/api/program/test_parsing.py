#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

import numpy as np

import piquasso as pq


def test_program_from_dict_with_StateVector_preparation():
    instructions_dict = {
        "instructions": [
            {
                "type": "StateVector",
                "attributes": {
                    "constructor_kwargs": {
                        "occupation_numbers": [1, 1],
                        "coefficient": 1.0,
                    },
                    "modes": [0, 1],
                },
            },
            {
                "type": "Beamsplitter",
                "attributes": {
                    "constructor_kwargs": {
                        "theta": np.pi / 3,
                        "phi": np.pi / 4,
                    },
                    "modes": [0, 1],
                },
            },
            {
                "type": "ParticleNumberMeasurement",
                "attributes": {
                    "constructor_kwargs": {},
                    "modes": [0],
                },
            },
        ]
    }

    program = pq.Program.from_dict(instructions_dict)

    assert isinstance(program.instructions[0], pq.StateVector)
    assert program.instructions[0].params == {
        "occupation_numbers": (1, 1),
        "coefficient": 1.0,
    }
    assert program.instructions[0].modes == [0, 1]

    assert isinstance(program.instructions[1], pq.Beamsplitter)
    assert program.instructions[1].params == {
        "theta": np.pi / 3,
        "phi": np.pi / 4,
    }
    assert program.instructions[1].modes == [0, 1]

    assert isinstance(program.instructions[2], pq.ParticleNumberMeasurement)
    assert program.instructions[2].params == {}
    assert program.instructions[2].modes == [0]


def test_program_from_dict_with_DensityMatrix_preparation():
    instructions_dict = {
        "instructions": [
            {
                "type": "DensityMatrix",
                "attributes": {
                    "constructor_kwargs": {
                        "bra": [1, 1],
                        "ket": [0, 0],
                        "coefficient": 1.0,
                    },
                    "modes": [0, 1],
                },
            },
            {
                "type": "Beamsplitter",
                "attributes": {
                    "constructor_kwargs": {
                        "theta": np.pi / 3,
                        "phi": np.pi / 4,
                    },
                    "modes": [0, 1],
                },
            },
            {
                "type": "ParticleNumberMeasurement",
                "attributes": {
                    "constructor_kwargs": {},
                    "modes": [0],
                },
            },
        ]
    }

    program = pq.Program.from_dict(instructions_dict)

    assert isinstance(program.instructions[0], pq.DensityMatrix)
    assert program.instructions[0].params == {
        "bra": (1, 1),
        "ket": (0, 0),
        "coefficient": 1.0,
    }
    assert program.instructions[0].modes == [0, 1]

    assert isinstance(program.instructions[1], pq.Beamsplitter)
    assert program.instructions[1].params == {
        "theta": np.pi / 3,
        "phi": np.pi / 4,
    }
    assert program.instructions[1].modes == [0, 1]

    assert isinstance(program.instructions[2], pq.ParticleNumberMeasurement)
    assert program.instructions[2].params == {}
    assert program.instructions[2].modes == [0]


def test_program_from_dict_from_external_instruction():
    class FakeInstruction(pq.Instruction):
        def __init__(self, first_param, second_param):
            super().__init__(
                params=dict(
                    first_param=first_param,
                    second_param=second_param,
                ),
            )

    instructions_dict = {
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

    program = pq.Program.from_dict(instructions_dict)

    assert isinstance(program.instructions[0], FakeInstruction)
    assert program.instructions[0].params == {
        "first_param": "first_param_value",
        "second_param": "second_param_value",
    }
    assert program.instructions[0].modes == ["some", "modes"]

    assert isinstance(program.instructions[1], FakeInstruction)
    assert program.instructions[1].params == {
        "first_param": "2nd_instructions_1st_param_value",
        "second_param": "2nd_instructions_2nd_param_value",
    }
    assert program.instructions[1].modes == ["some", "other", "modes"]
