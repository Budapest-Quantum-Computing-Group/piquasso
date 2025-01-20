#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

import numpy as np
import piquasso as pq

for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_density_matrix_StateVector_ordering(connector):
    d = 3

    state_vectors = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    for i in range(len(state_vectors)):
        with pq.Program() as preparation:
            pq.Q(0, 1, 2) | pq.StateVector(state_vectors[i])

        state = simulator.execute(preparation).state

        density_matrix = state.density_matrix

        assert np.isclose(density_matrix[i, i], 1.0)


@for_all_connectors
def test_StateVector_raises_InvalidParameter_for_invalid_occupation_numbers(connector):
    d = 3

    simulator = pq.fermionic.PureFockSimulator(d=d, connector=connector)

    with pq.Program() as preparation:
        pq.Q(0, 1, 2) | pq.StateVector([1, 2])

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.execute(preparation)

    assert error.value.args[0] == (
        "Invalid initial state specified: "
        "instruction="
        "StateVector(occupation_numbers=(1, 2), coefficient=1.0, modes=(0, 1, 2))"
    )
