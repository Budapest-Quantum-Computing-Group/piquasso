#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from pytest_lazy_fixtures import lf

import numpy as np

import networkx as nx


from piquasso._math.linalg import is_unitary
from piquasso._math.decompositions import (
    takagi,
)

from piquasso._simulators.connectors import (
    NumpyConnector,
    JaxConnector,
)


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_real_symmetric_2_by_2_matrix(connector):
    matrix = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=complex,
    )

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_complex_symmetric_2_by_2_matrix_with_multiplicities(connector):
    matrix = np.array(
        [
            [1, 2j],
            [2j, 1],
        ],
        dtype=complex,
    )

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_real_symmetric_3_by_3_matrix(connector):
    matrix = np.array(
        [
            [1, 2, 3],
            [2, 1, 5],
            [3, 5, 9],
        ],
        dtype=complex,
    )

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_complex_symmetric_3_by_3_matrix(connector):
    matrix = np.array(
        [
            [1, 2, 3j],
            [2, 1, 5j],
            [3j, 5j, 9],
        ],
    )

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.monkey
@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_complex_symmetric_6_by_6_matrix_with_multiplicities(
    connector,
    generate_unitary_matrix,
):
    singular_values = np.array([1, 1, 2, 2, 2, 3], dtype=complex)

    unitary = generate_unitary_matrix(6)

    matrix = unitary @ np.diag(singular_values) @ unitary.transpose()

    calculated_singular_values, calculated_unitary = takagi(matrix, connector)

    assert is_unitary(calculated_unitary)
    assert np.allclose(np.abs(calculated_singular_values), calculated_singular_values)
    assert np.allclose(
        matrix,
        calculated_unitary
        @ np.diag(calculated_singular_values)
        @ calculated_unitary.transpose(),
    )


@pytest.mark.monkey
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_complex_symmetric_N_by_N_matrix(
    N, connector, generate_complex_symmetric_matrix
):
    matrix = generate_complex_symmetric_matrix(N)
    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(matrix, unitary @ np.diag(singular_values) @ unitary.transpose())


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_adjacency_matrix_with_multiplicity(connector):
    adjacency_matrix = np.array(
        [
            [0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0],
        ],
        dtype=complex,
    )

    singular_values, unitary = takagi(adjacency_matrix, connector)
    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(
        adjacency_matrix, unitary @ np.diag(singular_values) @ unitary.transpose()
    )


@pytest.mark.parametrize(
    "connector", [NumpyConnector(), JaxConnector(), lf("tensorflow_connector")]
)
def test_takagi_on_random_adjacency_matrix_with_multiplicity(connector):
    graph = nx.erdos_renyi_graph(10, p=0.5)

    adjacency_matrix = nx.adjacency_matrix(graph).todense().astype(complex)

    singular_values, unitary = takagi(adjacency_matrix, connector)
    assert is_unitary(unitary)
    assert np.allclose(np.abs(singular_values), singular_values)
    assert np.allclose(
        adjacency_matrix, unitary @ np.diag(singular_values) @ unitary.transpose()
    )

