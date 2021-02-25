#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.state import State
from piquasso.backend import Backend
from piquasso.operations import Operation, ModelessOperation
from piquasso.program import Program


class TestProgramBase:
    @pytest.fixture
    def FakeBackend(self):
        class _FakeBackend(Backend):
            dummy_operation = Mock(name="dummy_operation")

            dummy_modeless_operation = Mock(name="dummy_modeless_operation")

        return _FakeBackend

    @pytest.fixture
    def FakeState(self, FakeBackend):
        class _FakeState(State):
            _backend_class = FakeBackend

        return _FakeState

    @pytest.fixture(autouse=True)
    def setup(self, FakeState):
        self.state = FakeState()

        self.program = Program(state=self.state)

    @pytest.fixture
    def DummyOperation(self, FakeBackend):
        class _DummyOperation(Operation):
            backends = {
                FakeBackend: FakeBackend.dummy_operation,
            }

        return _DummyOperation

    @pytest.fixture
    def DummyModelessOperation(self, FakeBackend):
        class _DummyModelessOperation(ModelessOperation):
            backends = {
                FakeBackend: FakeBackend.dummy_modeless_operation,
            }

        return _DummyModelessOperation
