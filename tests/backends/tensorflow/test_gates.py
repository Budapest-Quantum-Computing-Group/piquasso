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

import piquasso as pq
import pytest
import tensorflow as tf

import numpy as np


@pytest.mark.monkey
def test_Interferometer_numpy_array_as_parameter(generate_unitary_matrix):
    r = tf.Variable(0.01)
    d = 5
    interferometer = generate_unitary_matrix(d)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=3), calculator=pq.TensorflowCalculator()
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r)

        pq.Q(all) | pq.Interferometer(interferometer)

    simulator.execute(program)


def test_float32_dtype_calculations():
    inputs = np.array([0, 1, 2, 3, 4])
    d = 5
    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=10, normalize=False, dtype=np.float32),
        calculator=pq.TensorflowCalculator(
            decorate_with=tf.function(jit_compile=False)
        ),
    )

    preparation = [pq.Vacuum()]
    for j in range(d):
        r = tf.gather(inputs, j)
        preparation.append(
            pq.Squeezing(np.float32(0.5), np.float32(np.pi / 2)).on_modes(j)
        )
        preparation.append(pq.Displacement(r, np.float32(0)).on_modes(j))

    program = pq.Program(instructions=preparation)

    simulator.execute(program).state


def test_float32_dtype_calculations_complex():
    d = 5
    cvqnn_weights = tf.Variable(
        pq.cvqnn.generate_random_cvqnn_weights(layer_count=1, d=d),
        dtype=tf.float32
    )

    inputs = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=10, normalize=False, dtype=np.float32),
        calculator=pq.TensorflowCalculator(
            decorate_with=tf.function(jit_compile=False)
        ),
    )

    cvqnn = pq.cvqnn.create_layers(cvqnn_weights)

    preparation = [pq.Vacuum()]
    for j in range(d):
        r = tf.gather(inputs, j)
        preparation.append(
            pq.Squeezing(np.float32(0.5), np.float32(np.pi / 2)).on_modes(j)
        )
        preparation.append(pq.Displacement(np.float32(r), np.float32(0)).on_modes(j))

    program = pq.Program(instructions=preparation + cvqnn.instructions)

    simulator.execute(program).state