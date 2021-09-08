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
        def __init__(self, **params):
            super().__init__(params=params)

    class FakeState(pq.State):
        _instruction_map = {
            "DummyInstruction": "dummy_instruction",
        }

        dummy_instruction = Mock(name="dummy_instruction")

        d = 42

        def get_particle_detection_probability(self, occupation_number: tuple) -> float:
            raise NotImplementedError

    class FakePlugin(pq.Plugin):
        classes = {
            "FakeState": FakeState,
            "DummyInstruction": DummyInstruction,
        }

    pq.use(FakePlugin)
