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


def func(cutoff: int):
    matrix = tf.TensorArray(
        tf.complex128,
        size=0,
        dynamic_size=True,
        infer_shape=False,
    )
    matrix = matrix.write(0, np.arange(1))

    for i in tf.range(1, cutoff):
        matrix = matrix.write(i, np.arange(i + 1))

    return matrix


if __name__ == "__main__":
    decorator = tf.function(jit_compile=True)

    cutoff = 5

    enhanced = func

    enhanced = decorator(func)

    r = tf.Variable(0.1)
    phi = tf.Variable(0.1)

    value = enhanced(cutoff)

    print(value)

    print("GRAPH SIZE:", measure_graph_size(enhanced, cutoff))

    breakpoint()
