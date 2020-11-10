#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.context import Context
from piquasso.backend import Backend
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

        self.program = Program(
            state=Mock(name="State"),
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

    def test_program_instructions(self, DummyOperation):
        assert len(self.program.instructions) == 0

        with self.program:
            Q(0, 1) | DummyOperation(42) | DummyOperation(42, 320)

        self.program.execute()

        assert len(self.program.instructions) == 2
        self.program.backend.execute_instructions.assert_called_once()

    def test_modeless_program_instructions(self, DummyModelessOperation):
        with self.program:
            DummyModelessOperation(42)

        assert len(self.program.instructions) == 1


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

        assert len(self.program.instructions) == 2

        bs_gate = self.program.instructions[0]
        assert bs_gate["kwargs"]["modes"] == [1, 2]
        assert bs_gate["kwargs"]["params"] == [0.7853981633974483, 0]
        assert bs_gate["op"] == SamplingBackend.beamsplitter

        r_gate = self.program.instructions[1]
        assert r_gate["kwargs"]["modes"] == [1]
        assert r_gate["kwargs"]["params"] == [0.7853981633974483]
        assert r_gate["op"] == SamplingBackend.phaseshift
