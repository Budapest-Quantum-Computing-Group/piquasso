#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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


def test_state_collision_raises_RuntimeError(program):
    with pq.Program() as preparation:
        pq.Q() | pq.FakeState()

    with pytest.raises(RuntimeError) as error:
        with program:
            pq.Q() | preparation

    assert error.value.args[0] == (
        "The current program already has a state registered of type 'FakeState'."
    )
