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

from piquasso.api.calculator import Calculator

from piquasso._math.permanent import glynn_gray_permanent


class TensorflowCalculator(Calculator):
    def __init__(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "For this module of Piquasso, one needs to install tensorflow:"
                "\n"
                "pip install tensorflow"
            )

        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config

        np_config.enable_numpy_behavior()

        self._tnp = tnp
        self._tf = tf

        super().__init__(
            np=tnp,
            fallback_np=fallback_np,
            permanent_function=self._tf_glynn_grey_permanent,
            block_diag=self._block_diag,
            assign=self._assign,
            block=self._block,
            to_dense=self._to_dense,
            embed_in_identity=self._embed_in_identity,
            logm=self._logm,
            polar=self._polar,
            sqrtm=tf.linalg.sqrtm,
            svd=self._svd,
            expm=self._expm,
        )

    def _block_diag(self, *arrs):
        block_diagonalized = self._tf.linalg.LinearOperatorBlockDiag(
            [self._tf.linalg.LinearOperatorFullMatrix(arr) for arr in arrs]
        )

        return block_diagonalized.to_dense()

    def _tf_glynn_grey_permanent(self, matrix, rows, columns):
        return glynn_gray_permanent(matrix, rows, columns, np=self.np)

    def _assign(self, array, index, value):
        # NOTE: This is not as advanced as Numpy's indexing.

        state_vector_list = array.tolist()
        state_vector_list[index] = value

        return self.np.array(state_vector_list)

    def _block(self, arrays):
        # NOTE: This is not as advanced as `numpy.block`.

        d = len(arrays[0][0])

        output = []

        for i in range(d):
            output.append(self.np.concatenate([arrays[0][0][i], arrays[0][1][i]]))

        for i in range(d):
            output.append(self.np.concatenate([arrays[1][0][i], arrays[1][1][i]]))

        return self.np.stack(output)

    def _to_dense(self, index_map, dim):
        matrix = [[] for _ in range(dim)]

        for row in range(dim):
            for col in range(dim):
                index = (row, col)
                matrix[col].append(index_map.get(index, 0.0))

        for col in range(dim):
            matrix[col] = self.np.stack(matrix[col])

        return self.np.transpose(self.np.stack(matrix))

    def _embed_in_identity(self, matrix, indices, dim):
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

        return self._to_dense(index_map, dim)

    def _funm(self, matrix, func):
        eigenvalues, U = self._tf.linalg.eig(matrix)

        return U @ self.np.diag(func(eigenvalues)) @ self._tf.linalg.inv(U)

    def _logm(self, matrix):
        # NOTE: Tensorflow 2.0 has matrix logarithm, but it doesn't support gradient.
        # Therefore we had to implement our own.
        return self._funm(matrix, self.np.log)

    def _expm(self, matrix):
        # NOTE: Tensorflow 2.0 has matrix exponential, but it doesn't support gradient.
        # Therefore we had to implement our own.
        return self._funm(matrix, self.np.exp)

    def _polar(self, matrix, side="right"):
        P = self._tf.linalg.sqrtm(self.np.conj(matrix) @ matrix.T)
        Pinv = self._tf.linalg.inv(P)

        if side == "right":
            U = matrix @ Pinv
        elif side == "left":
            U = Pinv @ matrix
        else:
            raise ValueError(
                f"Argument 'side' should equal 'left' or 'right': side={side}"
            )

        return U, P

    def _svd(self, matrix):
        # NOTE: Tensorflow 2.0 SVD has different return tuple.

        S, V, W = self._tf.linalg.svd(matrix)

        return V, S, self.np.conj(W).T
