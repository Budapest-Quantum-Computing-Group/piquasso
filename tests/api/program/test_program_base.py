#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.api.circuit import Circuit
from piquasso.api.state import State
from piquasso.api.program import Program
from piquasso.api.operation import Operation


class TestProgramBase:
    @pytest.fixture
    def DummyOperation(self):
        class _DummyOperation(Operation):
            pass

        return _DummyOperation

    @pytest.fixture
    def FakeCircuit(self, DummyOperation):
        class _FakeCircuit(Circuit):
            dummy_operation = Mock(name="dummy_operation")

            def get_operation_map(self):
                return {
                    DummyOperation.__name__: self.dummy_operation,
                }

        return _FakeCircuit

    @pytest.fixture
    def FakeState(self, FakeCircuit):
        class _FakeState(State):
            circuit_class = FakeCircuit

        return _FakeState

    @pytest.fixture(autouse=True)
    def setup(self, FakeState):
        self.state = FakeState()

        self.program = Program(state=self.state)
