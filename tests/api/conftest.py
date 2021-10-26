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

import pytest

import piquasso as pq

from unittest.mock import Mock


@pytest.fixture
def FakePreparation():
    class FakePreparation(pq.Preparation):
        pass

    return FakePreparation


@pytest.fixture
def FakeGate():
    class FakeGate(pq.Gate):
        def __init__(self, **params):
            super().__init__(params=params)

    return FakeGate


@pytest.fixture
def FakeMeasurement():
    class FakeMeasurement(pq.Measurement):
        pass

    return FakeMeasurement


@pytest.fixture
def FakeState():
    class FakeState(pq.State):
        def __init__(self, d: int, config: pq.Config = None) -> None:
            super().__init__(config=config)

            self._d = d

        def get_particle_detection_probability(self, occupation_number: tuple) -> float:
            """
            NOTE: This needs to be here to be able to instantiate this class.
            """
            raise NotImplementedError

        @property
        def d(self):
            return self._d

        fake_preparation = Mock(name="fake_preparation")
        fake_gate = Mock(name="fake_gate")
        fake_measurement = Mock(name="fake_measurement")

    return FakeState


@pytest.fixture
def FakeSimulator(FakeState):
    class FakeSimulator(pq.Simulator):
        state_class = FakeState

        _instruction_map = {
            "FakePreparation": "fake_preparation",
            "FakeGate": "fake_gate",
            "FakeMeasurement": "fake_measurement",
        }

    return FakeSimulator
