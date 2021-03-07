#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.api.circuit import Circuit
from piquasso.api.state import State
from piquasso.api.program import Program
from piquasso.operations import Operation, ModelessOperation


class TestProgramBase:
    @pytest.fixture
    def DummyOperation(self):
        class _DummyOperation(Operation):
            pass

        return _DummyOperation

    @pytest.fixture
    def DummyModelessOperation(self):
        class _DummyModelessOperation(ModelessOperation):
            pass

        return _DummyModelessOperation

    @pytest.fixture
    def FakeCircuit(self, DummyOperation, DummyModelessOperation):
        class _FakeCircuit(Circuit):
            dummy_operation = Mock(name="dummy_operation")

            dummy_modeless_operation = Mock(name="dummy_modeless_operation")

            def get_operation_map(self):
                return {
                    DummyOperation.__name__: self.dummy_operation,
                    DummyModelessOperation.__name__: self.dummy_modeless_operation,
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
