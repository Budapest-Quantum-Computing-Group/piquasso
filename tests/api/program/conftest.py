#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from unittest.mock import Mock

import pytest

import piquasso as pq


@pytest.fixture(autouse=True)
def setup_plugin():
    class DummyInstruction(pq.Instruction):
        pass

    class FakeCircuit(pq.Circuit):
        dummy_instruction = Mock(name="dummy_instruction")

        def get_instruction_map(self):
            return {
                DummyInstruction.__name__: self.dummy_instruction,
            }

    class FakeState(pq.State):
        circuit_class = FakeCircuit
        d = 42

    class FakePlugin(pq.Plugin):
        classes = {
            "FakeState": FakeState,
            "FakeCircuit": FakeCircuit,
            "DummyInstruction": DummyInstruction,
        }

    pq.use(FakePlugin)


@pytest.fixture
def program():
    return pq.Program(state=pq.FakeState())
