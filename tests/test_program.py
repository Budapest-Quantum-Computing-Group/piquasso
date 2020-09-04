#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.context import Context
from piquasso.fock.backend import FockBackend
from piquasso.gates import B
from piquasso.mode import Q
from piquasso.program import Program


class TestProgram:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.program = Program(
            state=Mock(name="State"),
            backend_class=lambda _: Mock(FockBackend, name="Backend")
        )

    def test_current_program_in_program_context(self):
        with self.program:
            assert Context.current_program is self.program

        assert Context.current_program is None

    def test_program_instructions(self):
        assert len(self.program.instructions) == 0

        with self.program:
            Q(0, 1) | B(0.1, 0.4) | B(0.5, 0.3)

        self.program.execute()

        assert len(self.program.instructions) == 2
        self.program.backend.execute_instructions.assert_called_once()
