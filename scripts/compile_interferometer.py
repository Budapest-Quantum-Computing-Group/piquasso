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


np_config.enable_numpy_behavior()


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def func(interferometer):
    dtype = interferometer.dtype
    subspace_representations = tf.TensorArray(
        dtype=dtype,
        size=2,
        dynamic_size=True,
        infer_shape=False,
    )

    subspace_representations = subspace_representations.write(
        0, (tnp.identity(1) @ tnp.array([1.0], dtype=dtype))
    )
    subspace_representations = subspace_representations.write(
        1, (interferometer @ tnp.array([1.0, 0.0], dtype=dtype))
    )

    return subspace_representations


@tf.function
def trace(ta):
    size = ta.size
    ta2 = tf.TensorArray(dtype=ta.dtype, size=size)

    for i in tf.range(size):
        ta2 = ta2.write(i, ta.read(i)[0, 0])

    return ta2.stack()


if __name__ == "__main__":
    decorator = tf.function(jit_compile=True)

    cutoff = 5
    d = 2

    from scipy.stats import unitary_group

    U = tf.Variable(unitary_group.rvs(d))

    compiled = decorator(func)

    func(U)

    print()

    ta = compiled(U)

    print(ta)
