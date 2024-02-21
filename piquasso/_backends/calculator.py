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

    def accumulator(self, dtype, size, **kwargs):
        return []

    def write(self, accumulator, index, value):
        accumulator.append(value)

        return accumulator

    def stack_accumulator(self, accumulator):
        return self.forward_pass_np.stack(accumulator)

    def read(self, accumulator, index):
        return accumulator[index]

    def size(self, accumulator):
        return len(accumulator)


def _noop_custom_gradient(func):
    def noop_grad(*args, **kwargs):
        result, _ = func(*args, **kwargs)
        return result

    return noop_grad


class NumpyCalculator(_BuiltinCalculator):
    """The calculations for a simulation using NumPy (and SciPy).

    This is enabled by default in the built-in simulators.
    """

    def __init__(self):
        self.np = np
        self.fallback_np = np
        self.forward_pass_np = np
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

        self.custom_gradient = _noop_custom_gradient

        self.range = range

    def preprocess_input_for_custom_gradient(self, value):
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

    def __init__(self, no_custom_gradient=False):
        """
        Args:
            no_custom_gradient (bool, optional): Executes the forward pass
                using Tensorflow, instead of NumPy. This is sometimes necessary, e.g.,
                when one wants to decorate Piquasso code with
                `tf.function(jit_compile=True)`. In fact, when eager execution is
                disabled in Tensorflow, this argument is automatically set to `True`.
                Otherwise, it defaults to False.

        Raises:
            ImportError: When TensorFlow is not available.
        """
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

        no_custom_gradient = no_custom_gradient or not tf.executing_eagerly()
        self.no_custom_gradient = no_custom_gradient
        self.forward_pass_np = tnp if no_custom_gradient else np
        self.custom_gradient = (
            _noop_custom_gradient if no_custom_gradient else tf.custom_gradient
        )

        self.sqrtm = tf.linalg.sqrtm

        self.range = tf.range

    def _preprocess_input_for_custom_gradient_single(self, value):
        return value.numpy() if self._tf.is_tensor(value) else value

    def preprocess_input_for_custom_gradient(self, value):
        if self.no_custom_gradient:
            return value

        if isinstance(value, list):
            return [
                self._preprocess_input_for_custom_gradient_single(element)
                for element in value
            ]

        return self._preprocess_input_for_custom_gradient_single(value)

    def block_diag(self, *arrs):
        block_diagonalized = self._tf.linalg.LinearOperatorBlockDiag(
            [self._tf.linalg.LinearOperatorFullMatrix(arr) for arr in arrs]
        )

        return block_diagonalized.to_dense()

    def permanent(self, matrix, rows, columns):
        return glynn_gray_permanent(matrix, rows, columns, np=self.np)

    def assign(self, array, index, value):
        """
        NOTE: This method is very limited, and is a bit hacky, since TF does not support
        item assignment through its NumPy API.
        """

        if isinstance(array, self.fallback_np.ndarray):
            array[index] = value

            return array

        if isinstance(index, int):
            return self._tf.tensor_scatter_nd_update(array, [[index]], [value])

        # NOTE: When using `tf.function`, TensorFlow threw the following error:
        #
        # TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have types [int32, int64] that don't all match.  # noqa: E501
        #
        # To make it disappear, I had to convert all the indices to `int32`.
        index = index.astype(self.fallback_np.int32)

        if len(array.shape) == 1:
            return self._tf.tensor_scatter_nd_update(
                array, index.reshape(-1, 1), value.reshape(-1)
            )

        number_of_batches = array.shape[1]
        int_dtype = index.dtype

        flattened_index = index.reshape(-1)

        indices = self.fallback_np.column_stack(
            [
                self.fallback_np.tile(flattened_index, number_of_batches),
                self.fallback_np.concatenate(
                    [
                        self.fallback_np.full(len(flattened_index), i, dtype=int_dtype)
                        for i in range(number_of_batches)
                    ]
                ),
            ]
        )

        values = self.np.concatenate(
            [value[:, :, i].reshape(-1) for i in range(number_of_batches)]
        )

        return self._tf.tensor_scatter_nd_update(
            array,
            indices,
            values,
        )

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

    def accumulator(self, dtype, size, **kwargs):
        if not self.no_custom_gradient:
            return super().accumulator(dtype, size, **kwargs)

        return self._tf.TensorArray(dtype=dtype, size=size, **kwargs)

    def write(self, accumulator, index, value):
        if not self.no_custom_gradient:
            return super().write(accumulator, index, value)

        return accumulator.write(index, value)

    def stack_accumulator(self, accumulator):
        if not self.no_custom_gradient:
            return super().stack_accumulator(accumulator)

        return accumulator.stack()

    def read(self, accumulator, index):
        if not self.no_custom_gradient:
            return super().read(accumulator, index)

        return accumulator.read(index)

    def size(self, accumulator):
        if not self.no_custom_gradient:
            return super().size(accumulator)

        return accumulator.size()


class JaxCalculator(_BuiltinCalculator):
    """The calculations for a simulation using JAX."""

    def __init__(self):
        try:
            import jax.numpy as jnp
            import jax
        except ImportError:
            raise ImportError("You have invoked a feature which requires 'jax'.")

        self.np = jnp
        self.fallback_np = np
        self.forward_pass_np = jnp
        self.block_diag = jax.scipy.linalg.block_diag
        self.block = jnp.block
        self.logm = partial(jax.scipy.linalg.funm, func=jnp.log)
        self.expm = jax.scipy.linalg.expm
        self.powm = jnp.linalg.matrix_power
        self.polar = jax.scipy.linalg.polar
        self.sqrtm = jax.scipy.linalg.sqrtm
        self.svd = jnp.linalg.svd

        self.custom_gradient = _noop_custom_gradient

    def preprocess_input_for_custom_gradient(self, value):
        return value

    def assign(self, array, index, value):
        return array.at[index].set(value)

    def scatter(self, indices, updates, shape):
        embedded_matrix = self.np.zeros(shape, dtype=complex)
        indices_array = self.np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])

        embedded_matrix.at[composite_index].set(self.np.array(updates))

        return embedded_matrix
