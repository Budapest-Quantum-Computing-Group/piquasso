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

import numpy as np

import piquasso as pq

from piquasso import cvqnn


@pytest.mark.parametrize("layer_count, d", [(1, 2), (2, 1), (2, 2), (3, 4), (5, 10)])
def test_generate_random_cvqnn_weights(layer_count, d):
    weights = cvqnn.generate_random_cvqnn_weights(
        layer_count=layer_count, d=d, active_var=0.1, passive_var=0.1
    )

    assert weights.shape[0] == layer_count
    assert weights.shape[1] == 4 * d + 2 * (d * (d - 1) + max(1, d - 1))


def test_create_layers_yields_valid_program():
    d = 3

    weights = cvqnn.generate_random_cvqnn_weights(layer_count=10, d=d)
    layers = cvqnn.create_layers(weights)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 5)

        pq.Q() | layers

    simulator = pq.PureFockSimulator(d, config=pq.Config(cutoff=7))

    state = simulator.execute(program).state

    state.validate()


def test_create_program_yields_valid_program():
    d = 3

    weights = cvqnn.generate_random_cvqnn_weights(layer_count=10, d=d)
    program = cvqnn.create_program(weights)

    simulator = pq.PureFockSimulator(d, config=pq.Config(cutoff=7))

    state = simulator.execute(program).state

    state.validate()


def test_create_program_yields_valid_program_for_1_mode():
    d = 1

    weights = cvqnn.generate_random_cvqnn_weights(layer_count=10, d=d)
    program = cvqnn.create_program(weights)

    simulator = pq.PureFockSimulator(d, config=pq.Config(cutoff=7))

    state = simulator.execute(program).state

    state.validate()


def test_create_program_state_vector():
    d = 2

    weights = np.array(
        [
            [
                0.12856391,
                -0.03381019,
                -0.1377453,
                -0.00976438,
                -0.00987302,
                -0.03386338,
                -0.08224322,
                0.14441779,
                -0.00468172,
                -0.00120696,
                -0.04211308,
                -0.19803922,
                -0.0032648,
                0.01448497,
            ],
            [
                0.05684676,
                0.04780328,
                -0.0244914,
                0.01133317,
                0.00669194,
                0.02037788,
                0.03323016,
                0.13712651,
                -0.01981095,
                -0.00937427,
                0.00490652,
                -0.16186161,
                0.01149154,
                0.01250349,
            ],
            [
                0.14877177,
                0.04202048,
                0.16624834,
                0.00349866,
                -0.01133043,
                0.19474302,
                -0.01333464,
                -0.08560477,
                -0.00291201,
                -0.01066722,
                0.03029708,
                -0.06975938,
                0.00673455,
                -0.00111852,
            ],
        ]
    )

    program = cvqnn.create_program(weights)

    simulator = pq.PureFockSimulator(d)

    state = simulator.execute(program).state

    assert np.allclose(
        state.state_vector,
        [
            0.99927124 - 0.00000994j,
            -0.02215289 - 0.00339542j,
            -0.02876671 + 0.00127763j,
            -0.00304377 + 0.0010067j,
            -0.00414854 + 0.0005445j,
            0.00985761 + 0.00054058j,
            0.00013097 - 0.00000967j,
            0.00023877 - 0.00002112j,
            -0.00002112 - 0.00007153j,
            -0.00047556 - 0.00000858j,
        ],
    )


def test_create_program_with_invalid_weights():
    weights = np.empty(shape=(3, 7))

    with pytest.raises(pq.api.exceptions.CVQNNException) as error:
        cvqnn.create_program(weights)

    assert (
        error.value.args[0]
        == f"Invalid number of parameters specified: '{weights.shape[1]}'."
    )


@pytest.mark.parametrize(
    "number_of_parameters, number_of_modes", [(6, 1), (14, 2), (28, 3), (46, 4)]
)
def test_get_number_of_modes(number_of_parameters, number_of_modes):
    assert cvqnn.get_number_of_modes(number_of_parameters) == number_of_modes


def test_get_number_of_modes_with_invalid_number_of_parameters():
    invalid_number_of_parameters = 7

    with pytest.raises(pq.api.exceptions.CVQNNException) as error:
        cvqnn.get_number_of_modes(invalid_number_of_parameters)

    assert (
        error.value.args[0]
        == f"Invalid number of parameters specified: '{invalid_number_of_parameters}'."
    )


def test_get_cvqnn_weight_indices_1_mode():
    weight_indices = cvqnn.get_cvqnn_weight_indices(1)

    expected_weight_indices = [
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([3]),
        np.array([4]),
        np.array([5]),
    ]

    assert all(
        np.allclose(a, b) for a, b in zip(weight_indices, expected_weight_indices)
    )


def test_get_cvqnn_weight_indices():
    weight_indices = cvqnn.get_cvqnn_weight_indices(4)

    expected_weight_indices = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        np.array([15, 16, 17, 18]),
        np.array([19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]),
        np.array([34, 35, 36, 37]),
        np.array([38, 39, 40, 41]),
        np.array([42, 43, 44, 45]),
    ]

    assert all(
        np.allclose(a, b) for a, b in zip(weight_indices, expected_weight_indices)
    )
