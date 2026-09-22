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

import numpy as np
import piquasso as pq

from pytest_lazy_fixtures import lf

np.set_printoptions(precision=12)


@pytest.fixture
def tf_function(tf):
    return tf.function


@pytest.fixture
def tf_function_jit(tf):
    return tf.function(jit_compile=True)


def noop_decorator(func):
    return func


def test_tf_function_cvnn_layer_1_mode_1_layers(tf):
    d = 1
    cutoff = 3

    weights = tf.Variable(
        [[0.20961794, -0.00454663, 0.17257116, -0.00007423, -0.12339027, -0.01005965]],
        dtype=tf.float64,
    )

    @tf.function
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        -0.00014714134127510295,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                -0.0000007082,
                -0.0000000245,
                1.9822355012,
                -0.0000197403,
                -0.0000191909,
            ]
        ],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers(tf):
    d = 2
    cutoff = 3

    weights = tf.Variable(
        [
            [
                -0.01496882,
                -0.00769323,
                -0.23639019,
                0.00605814,
                -0.00198169,
                -0.00070044,
                -0.09768252,
                0.10580704,
                0.01289619,
                0.00523546,
                -0.1072419,
                -0.06125172,
                -0.00887781,
                -0.02219815,
            ],
            [
                -0.15894917,
                0.15323096,
                0.02459804,
                0.00319082,
                -0.0013642,
                0.06503896,
                -0.04608497,
                -0.04319411,
                -0.00326043,
                0.00621172,
                0.06301634,
                0.07470028,
                0.01229031,
                0.00481799,
            ],
        ]
    )

    @tf.function
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        0.01983292232816342,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.0003387847,
                0.000055788492,
                -0.000001310419,
                -0.000000000125,
                0.000000599267,
                1.9700215,
                0.17740987,
                0.0030813087,
                0.00035073576,
                0.0031765276,
                0.0003520168,
            ],
            [
                -0.0077148997,
                -0.00041063305,
                0.0034925407,
                -0.027122322,
                0.0004512317,
                -0.007975819,
                0.000059534465,
                0.0034475322,
                1.9940895,
                0.000007320457,
                0.0004920026,
                0.00000001988,
                0.004052103,
                -0.0,
            ],
        ],
    )


def test_tf_function_cvnn_layer_1_mode_1_layers_decorate_with_tf_function(tf):
    d = 1
    cutoff = 3

    weights = tf.Variable(
        [[0.20961794, -0.00454663, 0.17257116, -0.00007423, -0.12339027, -0.01005965]]
    )

    @tf.function
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(decorate_with=tf.function),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        -0.00014714134040471702,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                -0.000000708169,
                -0.000000024475,
                1.9822356,
                -0.000019740311,
                -0.000019190858,
            ]
        ],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers_decorate_with_tf_function(tf):
    d = 2
    cutoff = 3

    weights = tf.Variable(
        [
            [
                -0.01496882,
                -0.00769323,
                -0.23639019,
                0.00605814,
                -0.00198169,
                -0.00070044,
                -0.09768252,
                0.10580704,
                0.01289619,
                0.00523546,
                -0.1072419,
                -0.06125172,
                -0.00887781,
                -0.02219815,
            ],
            [
                -0.15894917,
                0.15323096,
                0.02459804,
                0.00319082,
                -0.0013642,
                0.06503896,
                -0.04608497,
                -0.04319411,
                -0.00326043,
                0.00621172,
                0.06301634,
                0.07470028,
                0.01229031,
                0.00481799,
            ],
        ]
    )

    @tf.function
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(decorate_with=tf.function),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        0.01983292232816342,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.0003387847,
                0.0000557885,
                -0.0000013104,
                -0.0000000001,
                0.0000005993,
                1.9700215,
                0.17740987,
                0.0030813087,
                0.0003507358,
                0.0031765276,
                0.0003520168,
            ],
            [
                -0.0077148997,
                -0.000410633,
                0.0034925407,
                -0.027122322,
                0.0004512317,
                -0.007975819,
                0.0000595345,
                0.0034475322,
                1.9940895,
                0.0000073205,
                0.0004920026,
                0.0000000199,
                0.004052103,
                -0.0,
            ],
        ],
    )


def test_tf_function_cvnn_layer_1_mode_1_layers_jit_compile(tf):
    d = 1
    cutoff = 3

    weights = tf.Variable(
        [[0.20961794, -0.00454663, 0.17257116, -0.00007423, -0.12339027, -0.01005965]]
    )

    @tf.function(jit_compile=True)
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(
                decorate_with=tf.function(jit_compile=True)
            ),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        -0.0001471413404047536,
    )
    assert np.allclose(
        mean_position_grad,
        [[0.0, -0.0000007082, -0.0000000245, 1.9822354, -0.0000197403, -0.0000191909]],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers_jit_compile(tf):
    d = 2
    cutoff = 3

    weights = tf.Variable(
        [
            [
                -0.01496882,
                -0.00769323,
                -0.23639019,
                0.00605814,
                -0.00198169,
                -0.00070044,
                -0.09768252,
                0.10580704,
                0.01289619,
                0.00523546,
                -0.1072419,
                -0.06125172,
                -0.00887781,
                -0.02219815,
            ],
            [
                -0.15894917,
                0.15323096,
                0.02459804,
                0.00319082,
                -0.0013642,
                0.06503896,
                -0.04608497,
                -0.04319411,
                -0.00326043,
                0.00621172,
                0.06301634,
                0.07470028,
                0.01229031,
                0.00481799,
            ],
        ]
    )

    @tf.function(jit_compile=True)
    def func(weights):
        simulator = pq.PureFockSimulator(
            d=d,
            config=pq.Config(cutoff=cutoff),
            connector=pq.TensorflowConnector(
                decorate_with=tf.function(jit_compile=True)
            ),
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_program(weights)

            state = simulator.execute(program).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(weights)

    assert np.allclose(
        mean_position,
        0.019832927857465484,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.0003387908,
                0.0000557884,
                -0.0000013104,
                -0.0000000001,
                0.0000005993,
                1.970022,
                0.17740993,
                0.0030813096,
                0.0003507358,
                0.0031765283,
                0.0003520169,
            ],
            [
                -0.0077149016,
                -0.0004106332,
                0.0034925423,
                -0.02712233,
                0.0004512318,
                -0.0079758195,
                0.0000595345,
                0.0034475331,
                1.9940897,
                0.0000073204,
                0.0004920026,
                0.0000000199,
                0.004052104,
                -0.0,
            ],
        ],
    )


@pytest.mark.parametrize(
    "decorator", (noop_decorator, lf("tf_function"), lf("tf_function_jit"))
)
def test_tf_function_cvnn_layer_1_mode_1_layers_custom_initial_state(decorator, tf):
    d = 1
    cutoff = 3

    weights = tf.Variable(
        [[0.20961794, -0.00454663, 0.17257116, -0.00007423, -0.12339027, -0.01005965]],
        dtype=np.float64,
    )

    initial_state_vector = tf.Variable([0.1, 0.3, 0.6], dtype=np.complex128)

    @decorator
    def func(initial_state_vector, weights):
        connector = pq.TensorflowConnector()
        config = pq.Config(cutoff=cutoff)

        initial_state = pq.PureFockState(d=1, connector=connector, config=config)

        initial_state.state_vector = initial_state_vector

        simulator = pq.PureFockSimulator(
            d=d,
            config=config,
            connector=connector,
        )

        with tf.GradientTape() as tape:
            program = pq.cvqnn.create_layers(weights)

            state = simulator.execute(program, initial_state=initial_state).state
            mean_position = state.mean_position(0)

        return mean_position, tape.gradient(mean_position, weights)

    mean_position, mean_position_grad = func(initial_state_vector, weights)

    assert np.allclose(
        mean_position,
        0.5329731259698964,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                -0.1973068264,
                0.2057361278,
                -0.1973790271,
                -1.2249751213,
                0.0000140195,
                -0.5483389353,
            ]
        ],
    )
