#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import json
import pytest
import collections

from piquasso import registry
from piquasso.context import Context
from piquasso.backend import Backend
from piquasso.state import State
from piquasso.mode import Q
from piquasso.program import Program
from piquasso.operations import Operation, ModelessOperation

from piquasso.sampling import SamplingBackend


class TestProgram:
    @pytest.fixture(autouse=True)
    def setup(self):
        class DummyBackend(Backend):
            def dummy_operation(self, modes, params):
                pass

            def dummy_modeless_operation(self, params):
                pass

        self.backend_class = DummyBackend

        self.state = Mock(name="DummyState")

        self.program = Program(
            state=self.state,
            backend_class=lambda state: Mock(self.backend_class, name="DummyBackend"),
        )

    @pytest.fixture
    def DummyOperation(self):
        class _DummyOperation(Operation):
            backends = {
                self.backend_class: self.backend_class.dummy_operation,
            }

        return _DummyOperation

    @pytest.fixture
    def DummyModelessOperation(self):
        class _DummyModelessOperation(ModelessOperation):
            backends = {
                self.backend_class: self.backend_class.dummy_modeless_operation,
            }

        return _DummyModelessOperation

    def test_current_program_in_program_context(self):
        with self.program:
            assert Context.current_program is self.program

        assert Context.current_program is None

    def test_program_operations(self, DummyOperation):
        assert len(self.program.operations) == 0

        with self.program:
            Q(0, 1) | DummyOperation(42) | DummyOperation(42, 320)

        self.program.execute()

        assert len(self.program.operations) == 2
        self.program.backend.execute_operations.assert_called_once()

    def test_modeless_program_operations(self, DummyModelessOperation):
        operation_parameter = 42
        with self.program:
            DummyModelessOperation(operation_parameter)

        assert len(self.program.operations) == 1
        assert self.program.operations[0].params == (operation_parameter,)

    def test_complex_program_stacking(self, DummyOperation, DummyModelessOperation):
        dummy_param1, dummy_param2, dummy_param3 = 1, 2, 3

        sub_program = Program()
        with sub_program:
            Q(0, 1) | DummyOperation(dummy_param1)
            Q(1, 2) | DummyOperation(dummy_param2)
            DummyModelessOperation(dummy_param3)

        with self.program:
            Q(0, 1, 2) | sub_program
            Q(1, 3, 2) | sub_program

        # Q(0, 1, 2) | sub_program
        assert self.program.operations[0].modes == (0, 1)
        assert self.program.operations[1].modes == (1, 2)
        assert self.program.operations[2].modes is None

        assert self.program.operations[0].params == (dummy_param1, )
        assert self.program.operations[1].params == (dummy_param2, )
        assert self.program.operations[2].params == (dummy_param3, )

        # Q(1, 3, 2) | sub_program
        assert self.program.operations[3].modes == (1, 3)
        assert self.program.operations[4].modes == (3, 2)
        assert self.program.operations[5].modes is None

        assert self.program.operations[3].params == (dummy_param1, )
        assert self.program.operations[4].params == (dummy_param2, )
        assert self.program.operations[5].params == (dummy_param3, )


def test_modeless_operation_is_called_on_execute():

    class DummyBackend(Backend):
        dummy_modeless_operation = Mock()

    class DummyModelessOperation(ModelessOperation):
        backends = {
            DummyBackend: DummyBackend.dummy_modeless_operation,
        }

    DummyState = collections.namedtuple("State", "d")

    number_of_modes = 420

    program = Program(
        state=DummyState(d=number_of_modes),
        backend_class=DummyBackend,
    )

    operation_parameter = 42

    with program:
        DummyModelessOperation(operation_parameter)

    program.execute()

    DummyBackend.dummy_modeless_operation.assert_called_once()


class TestProgramJSONParsing:
    @pytest.fixture
    def FakeState(self):
        class FakeState(State):
            def __init__(self, foo, bar, d):
                self.foo = foo
                self.bar = bar
                self.d = d

        return FakeState

    @pytest.fixture
    def FakeBackend(self):
        class FakeBackend(registry.ClassRecorder):
            def __init__(self, state):
                self.state = state

        return FakeBackend

    @pytest.fixture
    def FakeOperation(self):
        class FakeOperation(Operation):
            def __init__(self, params):
                self.params = params
                self.modes = None

        return FakeOperation

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
    def operations_mapping(self):
        return [
            {
                "type": "FakeOperation",
                "properties": {
                    "params": ["some", "params"],
                    "modes": ["some", "modes"],
                }
            },
            {
                "type": "FakeOperation",
                "properties": {
                    "params": ["some", "other", "params"],
                    "modes": ["some", "other", "modes"],
                }
            },
        ]

    def test_instantiation_using_mappings(
        self,
        FakeState,
        FakeBackend,
        FakeOperation,
        state_mapping,
        operations_mapping,
        number_of_modes,
    ):
        program = Program.from_properties(
            {
                "state": state_mapping,
                "backend_class": "FakeBackend",
                "operations": operations_mapping,
            }
        )

        assert program.state.foo == "fee"
        assert program.state.bar == "beer"
        assert program.state.d == number_of_modes

        assert type(program.backend) == FakeBackend

        assert program.operations[0].params == ["some", "params"]
        assert program.operations[0].modes == ["some", "modes"]
        assert program.operations[1].params == ["some", "other", "params"]
        assert program.operations[1].modes == ["some", "other", "modes"]

    def test_from_json(
        self,
        FakeState,
        FakeBackend,
        FakeOperation,
        state_mapping,
        operations_mapping,
        number_of_modes,
    ):
        json_ = json.dumps(
            {
                "state": state_mapping,
                "backend_class": "FakeBackend",
                "operations": operations_mapping,
            }
        )

        program = Program.from_json(json_)

        assert program.state.foo == "fee"
        assert program.state.bar == "beer"
        assert program.state.d == number_of_modes

        assert type(program.backend) == FakeBackend

        assert program.operations[0].params == ["some", "params"]
        assert program.operations[0].modes == ["some", "modes"]
        assert program.operations[1].params == ["some", "other", "params"]
        assert program.operations[1].modes == ["some", "other", "modes"]


class TestBlackbirdParsing:
    """
    TODO: Temporary solution to test `blackbird` code parsing.

    Ideally, `blackbird` parsing should be done `Backend`-independently.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.program = Program(
            state=Mock(name="State"),
            backend_class=SamplingBackend,
        )

    def test_from_blackbird(self):
        str = \
            """name StateTeleportation
            version 1.0

            BSgate(0.7853981633974483, 0) | [1, 2]
            Rgate(0.7853981633974483) | 1
            """
        self.program.loads_blackbird(str)

        assert len(self.program.operations) == 2

        bs_gate = self.program.operations[0]
        assert bs_gate.modes == [1, 2]
        assert bs_gate.params == (0, 0.7853981633974483)

        r_gate = self.program.operations[1]
        assert r_gate.modes == [1]
        assert r_gate.params == (0.7853981633974483,)
