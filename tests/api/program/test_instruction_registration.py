#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import piquasso as pq


def test_single_mode_single_instruction_registry(program):
    with program:
        pq.Q(0) | pq.DummyInstruction(dummyparam=420)

    assert len(program.instructions) == 1

    assert program.instructions[0].modes == (0,)
    assert program.instructions[0].params == {"dummyparam": 420}


def test_single_mode_multiple_instruction_registry(program):
    with program:
        pq.Q(0, 1) | pq.DummyInstruction(dummyparam=420) | pq.DummyInstruction(
            dummyparam1=42, dummyparam2=320
        )

    assert len(program.instructions) == 2

    assert program.instructions[0].modes == (0, 1)
    assert program.instructions[0].params == {"dummyparam": 420}

    assert program.instructions[1].modes == (0, 1)
    assert program.instructions[1].params == {
        "dummyparam1": 42,
        "dummyparam2": 320,
    }


def test_multiple_mode_single_instruction_registry(program):
    with program:
        pq.Q(2, 1, 0) | pq.DummyInstruction(dummyparam1=421)
        pq.Q(1) | pq.DummyInstruction(dummyparam2=1)
        pq.Q(0, 2) | pq.DummyInstruction(dummyparam3=999)

    assert len(program.instructions) == 3

    assert program.instructions[0].modes == (2, 1, 0)
    assert program.instructions[0].params == {"dummyparam1": 421}

    assert program.instructions[1].modes == (1,)
    assert program.instructions[1].params == {"dummyparam2": 1}

    assert program.instructions[2].modes == (0, 2)
    assert program.instructions[2].params == {"dummyparam3": 999}


def test_multiple_mode_multiple_instruction_registry(program):
    with program:
        pq.Q(4) | pq.DummyInstruction(param=2) | pq.DummyInstruction(param=0)
        pq.Q(0, 2) | pq.DummyInstruction(param=999)
        pq.Q(1, 0) | pq.DummyInstruction(param=1) | pq.DummyInstruction(param=9)

    assert len(program.instructions) == 5

    assert program.instructions[0].modes == (4,)
    assert program.instructions[0].params == {"param": 2}

    assert program.instructions[1].modes == (4,)
    assert program.instructions[1].params == {"param": 0}

    assert program.instructions[2].modes == (0, 2)
    assert program.instructions[2].params == {"param": 999}

    assert program.instructions[3].modes == (1, 0)
    assert program.instructions[3].params == {"param": 1}

    assert program.instructions[4].modes == (1, 0)
    assert program.instructions[4].params == {"param": 9}


def test_instruction_registration_with_no_modes_is_resolved_to_all_modes(
    program,
):
    with program:
        pq.Q() | pq.DummyInstruction(param="some-parameter")

    program.execute()

    assert program.instructions[0].modes == tuple(range(program.state.d))


def test_instruction_registration_with_all_keyword_is_resolved_to_all_modes(
    program,
):
    with program:
        pq.Q(all) | pq.DummyInstruction(param="some-parameter")

    program.execute()

    assert program.instructions[0].modes == tuple(range(program.state.d))
