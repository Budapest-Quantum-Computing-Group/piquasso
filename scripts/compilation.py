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

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config

import numpy as np

np_config.enable_numpy_behavior()


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def factorial(x):
    return tf.math.exp(tf.math.lgamma(tf.cast(x, tf.float32) + 1))


def create_single_mode_displacement_matrix(
    r: float,
    phi: float,
    cutoff: int,
):
    if tnp.isclose(r, 0.0):
        # NOTE: Tensorflow does not implement the NumPy API correctly, since in
        # tensorflow `np.power(0.0j, 0.0)` results in `nan+nanj`, whereas in NumPy it
        # is just 1. Instead of redefining `power` we just return when the squeezing
        # parameter is 0.
        return tnp.identity(cutoff, dtype=tf.complex128)

    cutoff_range = np.arange(cutoff)
    sqrt_indices = np.sqrt(cutoff_range)
    denominator = 1 / tnp.sqrt(factorial(cutoff_range))

    displacement = r * tnp.exp(1j * phi)
    displacement_conj = tnp.conj(displacement)

    matrix = tf.TensorArray(
        tf.complex128,
        size=cutoff,
    )
    previous = tnp.power(displacement, cutoff_range) * denominator
    matrix = matrix.write(0, previous)
    roll_index = tnp.arange(-1, cutoff - 1)

    for i in tf.range(1, cutoff):
        previous = sqrt_indices * previous[roll_index] - displacement_conj * previous
        matrix = matrix.write(i, previous)

    return tnp.exp(-0.5 * r**2) * matrix.stack() * denominator


if __name__ == "__main__":
    decorator = tf.function(jit_compile=True)

    cutoff = 5

    enhanced = create_single_mode_displacement_matrix

    enhanced = decorator(create_single_mode_displacement_matrix)

    r = tf.Variable(0.1)
    phi = tf.Variable(0.1)

    value = enhanced(r, phi, cutoff)

    print(value)

    print("GRAPH SIZE:", measure_graph_size(enhanced, r, phi, cutoff))

    breakpoint()
