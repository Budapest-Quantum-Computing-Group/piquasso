#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import json
import pytest

from piquasso import registry
from piquasso.state import State
from piquasso.program import Program
from piquasso.operations import Operation

from piquasso.sampling import SamplingBackend


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
            def __init__(self, first_param, second_param):
                super().__init__(first_param, second_param)

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
                    "params": {
                        "first_param": "first_param_value",
                        "second_param": "second_param_value",
                    },
                    "modes": ["some", "modes"],
                }
            },
            {
                "type": "FakeOperation",
                "properties": {
                    "params": {
                        "first_param": "2nd_operations_1st_param_value",
                        "second_param": "2nd_operations_2nd_param_value",
                    },
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

        assert program.operations[0].params == (
            "first_param_value", "second_param_value"
        )
        assert program.operations[0].modes == ["some", "modes"]

        assert program.operations[1].params == (
            "2nd_operations_1st_param_value", "2nd_operations_2nd_param_value"
        )
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

        assert program.operations[0].params == (
            "first_param_value", "second_param_value"
        )
        assert program.operations[0].modes == ["some", "modes"]
        assert program.operations[1].params == (
            "2nd_operations_1st_param_value", "2nd_operations_2nd_param_value"
        )
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
        string = \
            """name StateTeleportation
            version 1.0

            BSgate(0.7853981633974483, 0) | [1, 2]
            Rgate(0.7853981633974483) | 1
            """
        self.program.loads_blackbird(string)

        assert len(self.program.operations) == 2

        bs_gate = self.program.operations[0]
        assert bs_gate.modes == [1, 2]
        assert bs_gate.params == (0, 0.7853981633974483)

        r_gate = self.program.operations[1]
        assert r_gate.modes == [1]
        assert r_gate.params == (0.7853981633974483,)
