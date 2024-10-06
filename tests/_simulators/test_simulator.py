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


def test_GaussianSimulator_supports_NumpyConnector():
    pq.GaussianSimulator(d=1, connector=pq.NumpyConnector())


def test_GaussianSimulator_does_not_support_TensorflowConnector():
    connector = pq.TensorflowConnector()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        pq.GaussianSimulator(d=1, connector=connector)

    assert f"The connector '{connector}' is not supported." in error.value.args[0]


def test_SamplingSimulator_supports_NumpyConnector():
    pq.SamplingSimulator(d=1, connector=pq.NumpyConnector())


def test_SamplingSimulator_does_not_support_TensorflowConnector():
    connector = pq.TensorflowConnector()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        pq.SamplingSimulator(d=1, connector=connector)

    assert f"The connector '{connector}' is not supported." in error.value.args[0]


def test_FockSimulator_supports_NumpyConnector():
    pq.FockSimulator(d=1, connector=pq.NumpyConnector())


def test_FockSimulator_does_not_support_TensorflowConnector():
    connector = pq.TensorflowConnector()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        pq.FockSimulator(d=1, connector=connector)

    assert f"The connector '{connector}' is not supported." in error.value.args[0]


def test_PureFockSimulator_supports_NumpyConnector():
    pq.PureFockSimulator(d=1, connector=pq.NumpyConnector())


def test_PureFockSimulator_supports_TensorflowConnector():
    pq.PureFockSimulator(d=1, connector=pq.TensorflowConnector())
