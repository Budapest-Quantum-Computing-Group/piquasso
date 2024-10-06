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
import tensorflow as tf


def noop_decorator(func):
    return func


def test_tf_function_cvnn_layer_1_mode_1_layers():
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
        -0.00014718642085721634,
    )
    assert np.allclose(
        mean_position_grad,
        [[0.0, 0.00001151, -0.00000038, 1.9829729, -0.00001934, -0.00001949]],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers():
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
        0.019824614605943112,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.00177408,
                0.00011175,
                0.00000129,
                0.00000001,
                -0.00009018,
                1.96946914,
                0.17731823,
                0.00313927,
                0.00034899,
                0.00305477,
                0.00035107,
            ],
            [
                -0.00771081,
                -0.00041313,
                0.00346222,
                -0.02683549,
                0.00044953,
                -0.00797526,
                0.00006062,
                0.00336663,
                1.99442205,
                -0.00004225,
                0.00047722,
                -0.00000408,
                0.00381727,
                -0.00000682,
            ],
        ],
    )


def test_tf_function_cvnn_layer_1_mode_1_layers_decorate_with_tf_function():
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
        -0.00014718642085721634,
    )
    assert np.allclose(
        mean_position_grad,
        [[0.0, 0.00001151, -0.00000038, 1.9829729, -0.00001934, -0.00001949]],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers_decorate_with_tf_function():
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
        0.019824614605943112,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.00177408,
                0.00011175,
                0.00000129,
                0.00000001,
                -0.00009018,
                1.96946914,
                0.17731823,
                0.00313927,
                0.00034899,
                0.00305477,
                0.00035107,
            ],
            [
                -0.00771081,
                -0.00041313,
                0.00346222,
                -0.02683549,
                0.00044953,
                -0.00797526,
                0.00006062,
                0.00336663,
                1.99442205,
                -0.00004225,
                0.00047722,
                -0.00000408,
                0.00381727,
                -0.00000682,
            ],
        ],
    )


def test_tf_function_cvnn_layer_1_mode_1_layers_jit_compile():
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
        -0.00014718642085721634,
    )
    assert np.allclose(
        mean_position_grad,
        [[0.0, 0.00001151, -0.00000038, 1.9829729, -0.00001934, -0.00001949]],
    )


def test_tf_function_cvnn_layer_2_modes_2_layers_jit_compile():
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
        0.019824614605943112,
    )
    assert np.allclose(
        mean_position_grad,
        [
            [
                0.0,
                0.0,
                0.0,
                -0.00177408,
                0.00011175,
                0.00000129,
                0.00000001,
                -0.00009018,
                1.96946914,
                0.17731823,
                0.00313927,
                0.00034899,
                0.00305477,
                0.00035107,
            ],
            [
                -0.00771081,
                -0.00041313,
                0.00346222,
                -0.02683549,
                0.00044953,
                -0.00797526,
                0.00006062,
                0.00336663,
                1.99442205,
                -0.00004225,
                0.00047722,
                -0.00000408,
                0.00381727,
                -0.00000682,
            ],
        ],
    )


@pytest.mark.parametrize(
    "decorator", (noop_decorator, tf.function, tf.function(jit_compile=True))
)
def test_tf_function_cvnn_layer_1_mode_1_layers_custom_initial_state(decorator):
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
        0.28788324741568105,
    )
    assert np.allclose(
        mean_position_grad,
        [[-1.37846195, 0.13790631, -1.38045047, 0.27905517, 0.00014816, -2.28649567]],
    )
