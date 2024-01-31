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

import piquasso as pq


def test_single_instruction_program_stacking(FakeGate):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | FakeGate(param=420)

    with pq.Program() as program:
        pq.Q(0, 1) | sub_program

    assert program.instructions[0].modes == (0, 1)
    assert program.instructions[0].params == {"param": 420}


def test_multiple_instruction_program_stacking(FakeGate):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0) | FakeGate(param=2) | FakeGate(param=4)
        pq.Q(2, 3) | FakeGate(param=10)

    with pq.Program() as program:
        pq.Q(0, 1, 2, 3) | sub_program

    assert program.instructions[0].modes == (0,)
    assert program.instructions[0].params == {"param": 2}

    assert program.instructions[1].modes == (0,)
    assert program.instructions[1].params == {"param": 4}

    assert program.instructions[2].modes == (2, 3)
    assert program.instructions[2].params == {"param": 10}


def test_multiple_instruction_mixed_program_stacking(FakeGate):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | FakeGate(param=10)

    with pq.Program() as program:
        pq.Q(2) | FakeGate(param=2)
        pq.Q(0, 1) | sub_program
        pq.Q(3) | FakeGate(param=0)

    assert program.instructions[0].modes == (2,)
    assert program.instructions[0].params == {"param": 2}

    assert program.instructions[1].modes == (0, 1)
    assert program.instructions[1].params == {"param": 10}

    assert program.instructions[2].modes == (3,)
    assert program.instructions[2].params == {"param": 0}


def test_mixed_index_program_stacking(FakeGate):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | FakeGate(param=10)
        pq.Q(2, 3) | FakeGate(param=100)

    with pq.Program() as program:
        pq.Q(0, 2, 1, 3) | sub_program

    assert program.instructions[0].modes == (0, 2)
    assert program.instructions[0].params == {"param": 10}

    assert program.instructions[1].modes == (1, 3)
    assert program.instructions[1].params == {"param": 100}
