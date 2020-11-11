#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest
import collections

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

    def test_creating_program_using_string(self):
        program = Program(state=self.state, backend_class="DummyBackend")
        assert isinstance(program.backend, self.backend_class)

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
