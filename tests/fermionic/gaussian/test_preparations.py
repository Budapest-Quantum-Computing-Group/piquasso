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

from piquasso.api.exceptions import InvalidParameter


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_StateVector_raises_InvalidParameters_when_coefficient_is_not_1(connector):
    d = 3

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector) * 0.5

    with pytest.raises(InvalidParameter) as error:
        simulator.execute(program).state

    error_message = error.value.args[0]

    assert error_message == "Only 1.0 is permitted as coefficient for the state vector."


@for_all_connectors
def test_StateVector_raises_InvalidParameters_when_state_vector_is_not_0_or_1_vector(
    connector,
):
    d = 3

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    invalid_state_vector = [2, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(invalid_state_vector)

    with pytest.raises(InvalidParameter) as error:
        simulator.execute(program).state

    error_message = error.value.args[0]

    assert "Invalid initial state specified" in error_message
