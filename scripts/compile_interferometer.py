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
from piquasso._math.fock import cutoff_cardinality
from tensorflow.python.ops.numpy_ops import np_config

from piquasso._backends.fock.calculations import calculate_interferometer_helper_indices

np_config.enable_numpy_behavior()


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph

    return len(g.as_graph_def().node)


def func(interferometer, index_dict):
    cutoff = len(index_dict["subspace_index_tensor"]) + 2

    indices = [
        cutoff_cardinality(cutoff=n - 1, d=d) - cutoff_cardinality(cutoff=n - 2, d=d)
        for n in range(2, cutoff + 2)
    ]

    int_dtype = index_dict["subspace_index_tensor"][0].dtype

    indices_tensor = tf.TensorArray(int_dtype, size=cutoff, infer_shape=False)
    subspace_index_tensor = tf.TensorArray(int_dtype, size=cutoff, infer_shape=False)
    first_subspace_index_tensor = tf.TensorArray(
        int_dtype, size=cutoff, infer_shape=False
    )
    first_nonzero_index_tensor = tf.TensorArray(
        int_dtype, size=cutoff, infer_shape=False
    )
    sqrt_occupation_numbers_tensor = tf.TensorArray(
        tf.float64, size=cutoff, infer_shape=False
    )
    sqrt_first_occupation_numbers_tensor = tf.TensorArray(
        tf.float64, size=cutoff, infer_shape=False
    )

    for n in range(cutoff - 2):
        indices_tensor = indices_tensor.write(n, indices[n + 1])
        subspace_index_tensor = subspace_index_tensor.write(
            n, index_dict["subspace_index_tensor"][n]
        )
        first_subspace_index_tensor = first_subspace_index_tensor.write(
            n, index_dict["first_subspace_index_tensor"][n]
        )
        first_nonzero_index_tensor = first_nonzero_index_tensor.write(
            n, index_dict["first_nonzero_index_tensor"][n]
        )
        sqrt_occupation_numbers_tensor = sqrt_occupation_numbers_tensor.write(
            n, index_dict["sqrt_occupation_numbers_tensor"][n]
        )
        sqrt_first_occupation_numbers_tensor = (
            sqrt_first_occupation_numbers_tensor.write(
                n, index_dict["sqrt_first_occupation_numbers_tensor"][n]
            )
        )

    subspace_representations = tf.TensorArray(
        dtype=interferometer.dtype,
        size=cutoff,
        infer_shape=False,
        clear_after_read=False,
    )

    subspace_representations = subspace_representations.write(
        0, tf.reshape(tnp.identity(1, dtype=interferometer.dtype), [-1])
    )

    subspace_representations = subspace_representations.write(
        1, tf.reshape(interferometer, [-1])
    )

    for n in tf.range(2, cutoff):
        subspace_indices = subspace_index_tensor.read(n - 2)
        first_subspace_indices = first_subspace_index_tensor.read(n - 2)

        first_nonzero_indices = first_nonzero_index_tensor.read(n - 2)

        sqrt_occupation_numbers = sqrt_occupation_numbers_tensor.read(n - 2)
        sqrt_first_occupation_numbers = sqrt_first_occupation_numbers_tensor.read(n - 2)

        first_part_partially_indexed = tf.gather(interferometer, first_nonzero_indices)
        part = subspace_representations.read(n - 1)

        dim = indices_tensor.read(n - 2)

        part2 = tf.reshape(part, [dim, dim])

        second = part2[first_subspace_indices][:, subspace_indices]

        matrix = tnp.einsum(
            "ij,kj,kij->ik",
            sqrt_occupation_numbers,
            first_part_partially_indexed,
            second,
        )

        subspace_representations = subspace_representations.write(
            n,
            tf.reshape(
                tnp.transpose(matrix / sqrt_first_occupation_numbers).astype(
                    interferometer.dtype
                ),
                [-1],
            ),
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
    decorator = tf.function

    cutoff = 5
    d = 2

    index_dict = calculate_interferometer_helper_indices(d, cutoff)

    from scipy.stats import unitary_group

    U = tf.Variable(unitary_group.rvs(d))

    compiled = decorator(func)

    ta = func(U, index_dict)

    print()

    # ta = compiled(U, index_dict)

    # print(ta)
