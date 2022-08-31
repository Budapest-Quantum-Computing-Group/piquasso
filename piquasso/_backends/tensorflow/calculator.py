#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

import numpy as fallback_np

from piquasso.api.calculator import BaseCalculator

from piquasso._math.permanent import glynn_gray_permanent


class TensorflowCalculator(BaseCalculator):
    def __init__(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "You have invoked a feature which requires 'tensorflow'.\n"
                "You can install tensorflow via:\n"
                "\n"
                "pip install piquasso[tensorflow]"
            )

        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config

        np_config.enable_numpy_behavior()

        self._tf = tf
        self.np = tnp
        self.fallback_np = fallback_np
        self.sqrtm = tf.linalg.sqrtm

    def block_diag(self, *arrs):
        block_diagonalized = self._tf.linalg.LinearOperatorBlockDiag(
            [self._tf.linalg.LinearOperatorFullMatrix(arr) for arr in arrs]
        )

        return block_diagonalized.to_dense()

    def permanent(self, matrix, rows, columns):
        return glynn_gray_permanent(matrix, rows, columns, np=self.np)

    def assign(self, array, index, value):
        # NOTE: This is not as advanced as Numpy's indexing, only supports 1D arrays.

        state_vector_list = array.tolist()
        state_vector_list[index] = value

        return self.np.array(state_vector_list)

    def block(self, arrays):
        # NOTE: This is not as advanced as `numpy.block`, this function only supports
        # 4 same-length blocks.

        d = len(arrays[0][0])

        output = []

        for i in range(d):
            output.append(self.np.concatenate([arrays[0][0][i], arrays[0][1][i]]))

        for i in range(d):
            output.append(self.np.concatenate([arrays[1][0][i], arrays[1][1][i]]))

        return self.np.stack(output)

    def to_dense(self, index_map, dim):
        matrix = [[] for _ in range(dim)]

        for row in range(dim):
            for col in range(dim):
                index = (row, col)
                matrix[row].append(index_map.get(index, 0.0))

        return self.np.array(matrix)

    def embed_in_identity(self, matrix, indices, dim):
        index_map = {}

        small_dim = len(indices[0])
        for row in range(small_dim):
            for col in range(small_dim):
                index = (indices[0][row][col], indices[1][row][col])
                value = matrix[row, col]

                index_map[index] = value

        for index in range(dim):
            diagonal_index = (index, index)
            if diagonal_index not in index_map:
                index_map[diagonal_index] = 1.0

        return self.to_dense(index_map, dim)

    def logm(self, matrix):
        # NOTE: Tensorflow 2.0 has matrix logarithm, but it doesn't support gradient.
        # Therefore we had to implement our own.

        eigenvalues, U = self._tf.linalg.eig(matrix)

        log_eigenvalues = self.np.log(eigenvalues)

        return U @ self.np.diag(log_eigenvalues) @ self._tf.linalg.inv(U)

    def polar(self, matrix, side="right"):
        P = self._tf.linalg.sqrtm(self.np.conj(matrix) @ matrix.T)
        Pinv = self._tf.linalg.inv(P)

        if side == "right":
            U = matrix @ Pinv
        elif side == "left":
            U = Pinv @ matrix

        return U, P

    def svd(self, matrix):
        # NOTE: Tensorflow 2.0 SVD has different return tuple.

        S, V, W = self._tf.linalg.svd(matrix)

        return V, S, self.np.conj(W).T
