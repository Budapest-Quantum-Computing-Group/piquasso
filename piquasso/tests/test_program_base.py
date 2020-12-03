#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.backend import Backend
from piquasso.operations import Operation, ModelessOperation
from piquasso.program import Program


class TestProgramBase:
    @pytest.fixture(autouse=True)
    def setup(self):
        class DummyBackend(Backend):
            dummy_operation = Mock(name="dummy_operation")

            dummy_modeless_operation = Mock(name="dummy_modeless_operation")

        self.backend_class = DummyBackend

        self.state = Mock(name="DummyState")

        self.program = Program(
            state=self.state,
            backend_class=DummyBackend,
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

    @property
    def backend(self):
        return self.program.backend

    @property
    def operations(self):
        return self.program.operations
