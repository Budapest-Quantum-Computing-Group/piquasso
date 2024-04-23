#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

import numpy as np


@pytest.mark.parametrize(
    "calculator", [pq.NumpyCalculator(), pq.TensorflowCalculator(), pq.JaxCalculator()]
)
def test_multiply_passive_linear_gates(calculator):
    config = pq.Config(cutoff=5)
    beamsplitter1 = pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 5).on_modes(0, 1)
    phaseshifter = pq.Phaseshifter(phi=np.pi / 10).on_modes(1)
    beamsplitter2 = pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 7).on_modes(1, 3)

    instructions = [beamsplitter1, phaseshifter, beamsplitter2]

    interferometer = pq.utils.multiply_passive_linear_gates(
        instructions,
        calculator=calculator,
        config=config,
    )

    assert len(interferometer.modes) == 3

    preparation = [pq.StateVector([2, 1, 1, 0])]

    passive_gates_program = pq.Program(
        instructions=preparation + instructions,
    )

    interferometer_program = pq.Program(
        instructions=preparation + [interferometer],
    )

    simulator = pq.PureFockSimulator(d=4, calculator=calculator, config=config)

    passive_gates_state = simulator.execute(passive_gates_program).state

    interferometer_state = simulator.execute(interferometer_program).state

    assert passive_gates_state == interferometer_state
