#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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