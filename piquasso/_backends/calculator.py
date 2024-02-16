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

import scipy

import numpy as np

from functools import partial

from piquasso._math.permanent import np_glynn_gray_permanent, glynn_gray_permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from piquasso.api.calculator import BaseCalculator


class _BuiltinCalculator(BaseCalculator):
    """Base class for built-in calculators."""


class NumpyCalculator(_BuiltinCalculator):
    """The calculations for a simulation using NumPy (and SciPy).

    This is enabled by default in the built-in simulators.
    """

    def __init__(self):
        self.np = np
        self.fallback_np = np
        self.block_diag = scipy.linalg.block_diag
        self.block = np.block
        self.logm = scipy.linalg.logm
        self.expm = scipy.linalg.expm
        self.powm = np.linalg.matrix_power
        self.polar = scipy.linalg.polar
        self.sqrtm = scipy.linalg.sqrtm
        self.svd = np.linalg.svd

        self.permanent = np_glynn_gray_permanent
        self.hafnian = hafnian_with_reduction
        self.loop_hafnian = loop_hafnian_with_reduction

    def maybe_convert_to_numpy(self, value):
        return value

    def assign(self, array, index, value):
        array[index] = value

        return array

    def scatter(self, indices, updates, shape):
        embedded_matrix = np.zeros(shape, dtype=complex)
        indices_array = np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])

        embedded_matrix[composite_index] = np.array(updates)

        return embedded_matrix

    def custom_gradient(self, func):
        def wrapper(*args, **kwargs):
            result, _ = func(*args, **kwargs)
            return result

        return wrapper


class TensorflowCalculator(_BuiltinCalculator):
    """Calculator enabling calculating the gradients of certain instructions.

    This calculator is similar to
    :class:`~piquasso._backends.calculator.NumpyCalculator`, but it enables the
    simulator to use Tensorflow to be able to compute gradients.

    Note:
        Non-deterministic operations like
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement` are
        non-differentiable, please use a deterministic attribute of the resulting state
        instead.

    Example usage::

        import tensorflow as tf

        r = tf.Variable(0.43)

        tensorflow_calculator = pq.TensorflowCalculator()

        simulator = pq.PureFockSimulator(d=1, calculator=tensorflow_calculator)

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

        with tf.GradientTape() as tape:
            state = simulator.execute(program).state

            mean = state.mean_photon_number()

        gradient = tape.gradient(mean, [r])
    """

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
        self.fallback_np = np
        self.sqrtm = tf.linalg.sqrtm

    def maybe_convert_to_numpy(self, value):
        return value.numpy() if self._tf.is_tensor(value) else value

    def block_diag(self, *arrs):
        block_diagonalized = self._tf.linalg.LinearOperatorBlockDiag(
            [self._tf.linalg.LinearOperatorFullMatrix(arr) for arr in arrs]
        )

        return block_diagonalized.to_dense()

    def permanent(self, matrix, rows, columns):
        return glynn_gray_permanent(matrix, rows, columns, np=self.np)

    def assign(self, array, index, value):
        # NOTE: This is not as advanced as Numpy's indexing, only supports 1D arrays.

        return self._tf.tensor_scatter_nd_update(array, [[index]], [value])

    def block(self, arrays):
        # NOTE: This is not as advanced as `numpy.block`, this function only supports
        # 4 same-length blocks.

        return self._tf.concat(
            [self._tf.concat(arrays[0], 1), self._tf.concat(arrays[1], 1)], 0
        )

    def scatter(self, indices, updates, shape):
        return self._tf.scatter_nd(indices, updates, shape)

    def embed_in_identity(self, matrix, indices, dim):
        tf_indices = []
        updates = []

        small_dim = len(indices[0])
        for row in range(small_dim):
            for col in range(small_dim):
                index = [indices[0][row][col], indices[1][row][col]]
                update = matrix[row, col]

                tf_indices.append(index)
                updates.append(update)

        for index in range(dim):
            diagonal_index = [index, index]
            if diagonal_index not in tf_indices:
                tf_indices.append(diagonal_index)
                updates.append(1.0)

        return self.scatter(tf_indices, updates, (dim, dim))

    def _funm(self, matrix, func):
        eigenvalues, U = self._tf.linalg.eig(matrix)

        log_eigenvalues = func(eigenvalues)

        return U @ self.np.diag(log_eigenvalues) @ self._tf.linalg.inv(U)

    def logm(self, matrix):
        # NOTE: Tensorflow 2.0 has matrix logarithm, but it doesn't support gradient.
        # Therefore we had to implement our own.
        return self._funm(matrix, self.np.log)

    def expm(self, matrix):
        # NOTE: Tensorflow 2.0 has matrix exponential, but it doesn't support gradient.
        # Therefore we had to implement our own.
        return self._funm(matrix, self.np.exp)

    def powm(self, matrix, power):
        return self._funm(matrix, partial(self.np.power, x2=power))

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

    def custom_gradient(self, func):
        return self._tf.custom_gradient(func)
