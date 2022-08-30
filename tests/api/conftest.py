#
# Copyright 2021-2022 Budapest Quantum Computing Group
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
from piquasso.api.calculator import BaseCalculator


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
        def __init__(
            self, d: int, calculator: BaseCalculator, config: pq.Config = None
        ) -> None:
            super().__init__(calculator=calculator, config=config)

            self._d = d

        def get_particle_detection_probability(self, occupation_number: tuple) -> float:
            """
            NOTE: This needs to be here to be able to instantiate this class.
            """
            raise NotImplementedError

        @property
        def d(self):
            return self._d

        @property
        def fock_probabilities(self):
            return list()

        def validate(self):
            pass

    return FakeState


@pytest.fixture
def FakeConfig():
    class FakeConfig(pq.api.config.Config):
        pass

    return FakeConfig


@pytest.fixture
def FakeSimulator(FakeState, FakeConfig, FakePreparation, FakeGate, FakeMeasurement):
    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        return pq.api.result.Result(state=state)

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            FakePreparation: fake_calculation,
            FakeGate: fake_calculation,
            FakeMeasurement: fake_calculation,
        }

    return FakeSimulator
