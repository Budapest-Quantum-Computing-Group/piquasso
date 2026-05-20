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

import networkx as nx
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

import piquasso as pq
from piquasso._math.decompositions import (
    takagi,
    williamson,
)
from piquasso._math.linalg import is_diagonal, is_unitary
from piquasso._math.symplectic import is_symplectic, xp_symplectic_form
from piquasso._simulators.connectors import JaxConnector, NumpyConnector, TorchConnector

for_all_connectors = pytest.mark.parametrize(
    "connector",
    [NumpyConnector(), JaxConnector(), TorchConnector(), lf("tensorflow_connector")],
)


@for_all_connectors
def test_takagi_on_real_symmetric_2_by_2_matrix(connector):
    matrix = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=complex,
    )
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    matrix = connector.np.array(matrix)

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert np.allclose(connector.np.abs(singular_values), singular_values)
    # NOTE: We need matmul instead of @,
    # because torch.Tensors do not automatically promote types.
    assert np.allclose(
        matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@for_all_connectors
def test_takagi_on_complex_symmetric_2_by_2_matrix_with_multiplicities(connector):
    matrix = np.array(
        [
            [1, 2j],
            [2j, 1],
        ],
        dtype=complex,
    )
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    matrix = connector.np.array(matrix)

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert connector.np.allclose(connector.np.abs(singular_values), singular_values)
    # NOTE: We need matmul instead of @,
    # because torch.Tensors do not automatically promote types.
    assert connector.np.allclose(
        matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@for_all_connectors
def test_takagi_on_real_symmetric_3_by_3_matrix(connector):
    matrix = np.array(
        [
            [1, 2, 3],
            [2, 1, 5],
            [3, 5, 9],
        ],
        dtype=complex,
    )
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    matrix = connector.np.array(matrix)

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert connector.np.allclose(connector.np.abs(singular_values), singular_values)
    assert connector.np.allclose(
        matrix, np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T
    )


@for_all_connectors
def test_takagi_on_complex_symmetric_3_by_3_matrix(connector):
    matrix = np.array(
        [
            [1, 2, 3j],
            [2, 1, 5j],
            [3j, 5j, 9],
        ],
    )
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    matrix = connector.np.array(matrix)

    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert connector.np.allclose(connector.np.abs(singular_values), singular_values)
    # NOTE: We need matmul instead of @,
    # because torch.Tensors do not automatically promote types.
    assert connector.np.allclose(
        matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_takagi_on_complex_symmetric_6_by_6_matrix_with_multiplicities(
    connector,
    generate_unitary_matrix,
):
    singular_values = np.array([1, 1, 2, 2, 2, 3], dtype=complex)

    unitary = generate_unitary_matrix(6)

    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    unitary = connector.np.array(unitary)
    singular_values = connector.np.array(singular_values)

    matrix = (
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T
    )

    calculated_singular_values, calculated_unitary = takagi(matrix, connector)

    assert is_unitary(calculated_unitary)
    assert connector.np.allclose(
        connector.np.abs(calculated_singular_values), calculated_singular_values
    )
    assert connector.np.allclose(
        matrix,
        connector.np.matmul(
            calculated_unitary, connector.np.diag(calculated_singular_values)
        )
        @ calculated_unitary.T,
    )


@pytest.mark.monkey
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
@for_all_connectors
def test_takagi_on_complex_symmetric_N_by_N_matrix(
    N, connector, generate_complex_symmetric_matrix
):
    matrix = generate_complex_symmetric_matrix(N)
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    matrix = connector.np.array(matrix)
    singular_values, unitary = takagi(matrix, connector)

    assert is_unitary(unitary)
    assert connector.np.allclose(connector.np.abs(singular_values), singular_values)
    assert connector.np.allclose(
        matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@for_all_connectors
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
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    adjacency_matrix = connector.np.array(adjacency_matrix)

    singular_values, unitary = takagi(adjacency_matrix, connector)
    assert is_unitary(unitary)
    assert connector.np.allclose(np.abs(singular_values), singular_values)
    assert connector.np.allclose(
        adjacency_matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@for_all_connectors
def test_takagi_on_random_adjacency_matrix_with_multiplicity(connector):
    graph = nx.erdos_renyi_graph(10, p=0.5)

    adjacency_matrix = nx.adjacency_matrix(graph).todense().astype(complex)
    # TorchConnector operates on `torch.Tensor`, so we need to parse.
    adjacency_matrix = connector.np.array(adjacency_matrix)

    singular_values, unitary = takagi(adjacency_matrix, connector)
    assert is_unitary(unitary)
    assert connector.np.allclose(connector.np.abs(singular_values), singular_values)
    assert connector.np.allclose(
        adjacency_matrix,
        connector.np.matmul(unitary, connector.np.diag(singular_values)) @ unitary.T,
    )


@pytest.mark.parametrize("connector", [NumpyConnector(), JaxConnector()])
def test_williamson_with_identity(connector):
    covariance_matrix = np.identity(4)
    symplectic, diagonal = williamson(covariance_matrix, connector)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


@pytest.mark.parametrize("connector", [NumpyConnector(), JaxConnector()])
def test_williamson_with_diagonal_matrix(connector):
    covariance_matrix = np.diag([1, 2, 3, 4])
    symplectic, diagonal = williamson(covariance_matrix, connector)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


@pytest.mark.parametrize("connector", [NumpyConnector(), JaxConnector()])
def test_williamson_with_squeezed_covariance_matrix(connector):
    d = 3
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=np.pi / 3)
        pq.Q(1, 2) | pq.Squeezing2(r=0.2, phi=np.pi / 5)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    covariance_matrix = state.xpxp_covariance_matrix

    symplectic, diagonal = williamson(covariance_matrix, connector)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(covariance_matrix, symplectic @ diagonal @ symplectic.T)


@pytest.mark.monkey
@pytest.mark.parametrize("connector", [NumpyConnector(), JaxConnector()])
def test_williamson_with_random_positive_definite_matrix(
    generate_random_positive_definite_matrix, connector
):
    dim = 4
    matrix = generate_random_positive_definite_matrix(dim)

    symplectic, diagonal = williamson(matrix, connector)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(matrix, symplectic @ diagonal @ symplectic.T)


@pytest.mark.parametrize("connector", [NumpyConnector(), JaxConnector()])
def test_williamson_with_special_matrix(connector):
    """
    Sometimes scipy.linalg.sqrtm returns complex matrices even when real matrices with
    positive eigenvalues got specified. This test specifies a matrix for which the
    square root turns out to be complex even if it is real and positive definite.

    When a complex matrix is specified to the schur algorithm, automatically the complex
    schur algorithm is executed instead of the desired real schur algorithm.
    """
    matrix = np.array(
        [
            [
                1.8577097765113275,
                -0.03968353838339344,
                -0.13480023344314673,
                0.09038151421983714,
                -0.19128617316388555,
                0.061794725510946794,
                0.14126733590055737,
                -0.13966704763096188,
            ],
            [
                -0.03968353838339344,
                2.349487427421141,
                -0.07259190627416134,
                -0.01569343729182265,
                0.06179472551094684,
                0.05723660406987829,
                -0.07428264466311996,
                0.06923453621903403,
            ],
            [
                -0.13480023344314673,
                -0.07259190627416134,
                2.1222235731327235,
                0.21237578147380476,
                0.14126733590055734,
                -0.07428264466311996,
                0.09628621679115495,
                0.12278740167165433,
            ],
            [
                0.09038151421983714,
                -0.015693437291822643,
                0.21237578147380476,
                2.01240160261767,
                -0.13966704763096188,
                0.06923453621903401,
                0.12278740167165435,
                0.18926141391838203,
            ],
            [
                -0.19128617316388555,
                0.06179472551094682,
                0.14126733590055737,
                -0.13966704763096188,
                2.2225572459649756,
                0.03968353838339344,
                0.13480023344314673,
                -0.09038151421983714,
            ],
            [
                0.061794725510946814,
                0.05723660406987829,
                -0.07428264466311996,
                0.06923453621903403,
                0.03968353838339344,
                1.730779595055162,
                0.07259190627416134,
                0.015693437291822643,
            ],
            [
                0.14126733590055734,
                -0.07428264466311996,
                0.09628621679115495,
                0.12278740167165433,
                0.13480023344314673,
                0.07259190627416134,
                1.9580434493435799,
                -0.21237578147380476,
            ],
            [
                -0.13966704763096188,
                0.06923453621903401,
                0.12278740167165435,
                0.18926141391838203,
                -0.09038151421983714,
                0.015693437291822637,
                -0.21237578147380476,
                2.0678654198586335,
            ],
        ],
        dtype=float,
    )

    symplectic, diagonal = williamson(matrix, connector)

    assert is_diagonal(diagonal)
    assert is_symplectic(symplectic, form_func=xp_symplectic_form)
    assert np.all(np.isreal(symplectic))
    assert np.allclose(matrix, symplectic @ diagonal @ symplectic.T)
