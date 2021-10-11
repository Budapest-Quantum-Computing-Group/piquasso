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

from unittest.mock import Mock


@pytest.fixture
def FakeState():
    class FakeState(pq.State):
        def __init__(self, d: int, config: pq.Config = None) -> None:
            super().__init__(config=config)

            self._d = d

        def get_particle_detection_probability(self, occupation_number: tuple) -> float:
            """
            NOTE: This needs to be here to be able to instantiate this class.
            """
            raise NotImplementedError

        @property
        def d(self):
            return self._d

        dummy_instruction = Mock(name="dummy_instruction")

    return FakeState


@pytest.fixture
def FakeSimulator(FakeState):
    class FakeSimulator(pq.Simulator):
        _instruction_map = {
            "DummyInstruction": "dummy_instruction",
        }

        state_class = FakeState

    return FakeSimulator


def test_single_mode_single_instruction_registry(DummyInstruction):
    with pq.Program() as program:
        pq.Q(0) | DummyInstruction(dummyparam=420)

    assert len(program.instructions) == 1

    assert program.instructions[0].modes == (0,)
    assert program.instructions[0].params == {"dummyparam": 420}


def test_single_mode_multiple_instruction_registry(DummyInstruction):
    with pq.Program() as program:
        pq.Q(0, 1) | DummyInstruction(dummyparam=420) | DummyInstruction(
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


def test_multiple_mode_single_instructionregistry(DummyInstruction):
    with pq.Program() as program:
        pq.Q(2, 1, 0) | DummyInstruction(dummyparam1=421)
        pq.Q(1) | DummyInstruction(dummyparam2=1)
        pq.Q(0, 2) | DummyInstruction(dummyparam3=999)

    assert len(program.instructions) == 3

    assert program.instructions[0].modes == (2, 1, 0)
    assert program.instructions[0].params == {"dummyparam1": 421}

    assert program.instructions[1].modes == (1,)
    assert program.instructions[1].params == {"dummyparam2": 1}

    assert program.instructions[2].modes == (0, 2)
    assert program.instructions[2].params == {"dummyparam3": 999}


def test_multiple_mode_multiple_instructionregistry(DummyInstruction):
    with pq.Program() as program:
        pq.Q(4) | DummyInstruction(param=2) | DummyInstruction(param=0)
        pq.Q(0, 2) | DummyInstruction(param=999)
        pq.Q(1, 0) | DummyInstruction(param=1) | DummyInstruction(param=9)

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
    DummyInstruction,
    FakeSimulator,
):
    with pq.Program() as program:
        pq.Q() | DummyInstruction(param="some-parameter")

    simulator = FakeSimulator(d=42)
    state = simulator.execute(program).state

    assert program.instructions[0].modes == tuple(range(state.d))


def test_instruction_registration_with_all_keyword_is_resolved_to_all_modes(
    DummyInstruction,
    FakeSimulator,
):
    with pq.Program() as program:
        pq.Q(all) | DummyInstruction(param="some-parameter")

    simulator = FakeSimulator(d=42)
    state = simulator.execute(program).state

    assert program.instructions[0].modes == tuple(range(state.d))
