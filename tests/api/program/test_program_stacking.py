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


def test_single_instruction_program_stacking(program):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | pq.DummyInstruction(param=420)

    with program:
        pq.Q(0, 1) | sub_program

    assert program.instructions[0].modes == (0, 1)
    assert program.instructions[0].params == {"param": 420}


def test_multiple_instruction_program_stacking(program):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0) | pq.DummyInstruction(param=2) | pq.DummyInstruction(param=4)
        pq.Q(2, 3) | pq.DummyInstruction(param=10)

    with program:
        pq.Q(0, 1, 2, 3) | sub_program

    assert program.instructions[0].modes == (0,)
    assert program.instructions[0].params == {"param": 2}

    assert program.instructions[1].modes == (0,)
    assert program.instructions[1].params == {"param": 4}

    assert program.instructions[2].modes == (2, 3)
    assert program.instructions[2].params == {"param": 10}


def test_multiple_instruction_mixed_program_stacking(program):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | pq.DummyInstruction(param=10)

    with program:
        pq.Q(2) | pq.DummyInstruction(param=2)
        pq.Q(0, 1) | sub_program
        pq.Q(3) | pq.DummyInstruction(param=0)

    assert program.instructions[0].modes == (2,)
    assert program.instructions[0].params == {"param": 2}

    assert program.instructions[1].modes == (0, 1)
    assert program.instructions[1].params == {"param": 10}

    assert program.instructions[2].modes == (3,)
    assert program.instructions[2].params == {"param": 0}


def test_mixed_index_program_stacking(program):
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | pq.DummyInstruction(param=10)
        pq.Q(2, 3) | pq.DummyInstruction(param=100)

    with program:
        pq.Q(0, 2, 1, 3) | sub_program

    assert program.instructions[0].modes == (0, 2)
    assert program.instructions[0].params == {"param": 10}

    assert program.instructions[1].modes == (1, 3)
    assert program.instructions[1].params == {"param": 100}


def test_main_program_inherits_state(program):
    with pq.Program() as preparation:
        pq.Q() | pq.FakeState()

    with pq.Program() as main:
        pq.Q() | preparation

    assert main.state is not None
    assert main.state is not preparation.state, (
        "The state should be copied from the subprogram due to the state's "
        "mutability."
    )


def test_main_program_inherits_state_and_instructions(
    program
):
    with pq.Program() as preparation:
        pq.Q() | pq.FakeState()

        pq.Q(0, 1) | pq.DummyInstruction(param=10)
        pq.Q(2, 3) | pq.DummyInstruction(param=20)

    with pq.Program() as main:
        pq.Q(0, 1, 2, 3) | preparation

    assert main.state is not None
    assert main.state is not preparation.state, (
        "The state should be copied from the subprogram due to the state's "
        "mutability."
    )

    assert main.instructions[0] == pq.DummyInstruction(param=10).on_modes(0, 1)
    assert main.instructions[1] == pq.DummyInstruction(param=20).on_modes(2, 3)


def test_main_program_inherits_state_and_instructions_without_modes_specified(
    program
):
    state = pq.FakeState()
    with pq.Program() as preparation:
        pq.Q() | state

        pq.Q(0, 1) | pq.DummyInstruction(param=10)
        pq.Q(2, 3) | pq.DummyInstruction(param=20)

    with pq.Program() as main:
        first_instruction = pq.Q() | preparation

    assert first_instruction.modes == tuple(range(state.d))

    assert main.state is not None
    assert main.state is not preparation.state, (
        "The state should be copied from the subprogram due to the state's "
        "mutability."
    )

    assert main.instructions[0] == pq.DummyInstruction(param=10).on_modes(0, 1)
    assert main.instructions[1] == pq.DummyInstruction(param=20).on_modes(2, 3)


def test_state_collision_raises_InvalidProgram(program):
    with pq.Program() as preparation:
        pq.Q() | pq.FakeState()

    with pytest.raises(pq.api.errors.InvalidProgram) as error:
        with program:
            pq.Q() | preparation

    assert error.value.args[0] == (
        "The program already has a state registered of type 'FakeState'."
    )


def test_all_modes_program_stacking_in_sub_program():
    sub_program = pq.Program()
    with sub_program:
        pq.Q(all) | pq.DummyInstruction(param=10)

    with pq.Program() as program:
        pq.Q(1, 0, 2) | sub_program
        pq.Q(1, 2) | pq.DummyInstruction(param=100)

    assert program.instructions[0].modes == (1, 0, 2)
    assert program.instructions[0].params == {"param": 10}

    assert program.instructions[1].modes == (1, 2)
    assert program.instructions[1].params == {"param": 100}


def test_all_modes_program_stacking_in_main_program():
    sub_program = pq.Program()
    with sub_program:
        pq.Q(0, 1) | pq.DummyInstruction(param=10)
        pq.Q(2, 3) | pq.DummyInstruction(param=100)

    with pq.Program() as program:
        pq.Q(all) | sub_program
        pq.Q(0, 1) | pq.DummyInstruction(param=1000)

    assert program.instructions[0].modes == (0, 1)
    assert program.instructions[0].params == {"param": 10}

    assert program.instructions[1].modes == (2, 3)
    assert program.instructions[1].params == {"param": 100}

    assert program.instructions[2].modes == (0, 1)
    assert program.instructions[2].params == {"param": 1000}


def test_all_modes_program_stacking_in_both_sub_and_main_program():
    sub_program = pq.Program()
    with sub_program:
        pq.Q(all) | pq.DummyInstruction(param=10)
        pq.Q(1, 0) | pq.DummyInstruction(param=100)

    with pq.Program() as program:
        pq.Q(all) | sub_program
        pq.Q(0, 4) | pq.DummyInstruction(param=1000)

    assert len(program.instructions[0].modes) == 0
    assert program.instructions[0].params == {"param": 10}

    assert program.instructions[1].modes == (1, 0)
    assert program.instructions[1].params == {"param": 100}

    assert program.instructions[2].modes == (0, 4)
    assert program.instructions[2].params == {"param": 1000}
