#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

from piquasso.context import Context
from piquasso.sampling import SamplingBackend
from piquasso.operations import B, Sampling
from piquasso.mode import Q
from piquasso.program import Program


class TestProgram:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.program = Program(
            state=Mock(name="State"),
            backend_class=lambda _: Mock(SamplingBackend, name="Backend")
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

    def test_modeless_program_instructions(self):
        with self.program:
            Sampling(shots=10)

        assert len(self.program.instructions) == 1

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
