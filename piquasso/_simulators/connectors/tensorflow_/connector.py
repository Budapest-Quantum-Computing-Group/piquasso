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

import numpy as np

from functools import partial

from ..connector import BuiltinConnector


class TensorflowConnector(BuiltinConnector):
    """Connector enabling calculating the gradients of certain instructions.

    This connector is similar to
    :class:`~piquasso._simulators.connector.NumpyConnector`, but it enables the
    simulator to use Tensorflow to be able to compute gradients.

    Example usage::

        import piquasso as pq
        import tensorflow as tf

        r = tf.Variable(0.43)

        tensorflow_connector = pq.TensorflowConnector()

        simulator = pq.PureFockSimulator(d=1, connector=tensorflow_connector)

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

        with tf.GradientTape() as tape:
            state = simulator.execute(program).state

            mean = state.mean_photon_number()

        gradient = tape.gradient(mean, [r])

    Note:
        Non-deterministic operations like
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement` are
        non-differentiable, please use a deterministic attribute of the resulting state
        instead.

    """

    def __init__(self, decorate_with=None):
        """
        Args:
            decorate_with (function, optional): A function to decorate calculations
                with. Currently, only `tf.function` is supported. Specifying this may
                reduce runtime after the tracing step. See
                `Better performance with tf.function <https://www.tensorflow.org/guide/function>`_.

        Raises:
            ImportError: When TensorFlow is not available.
        """  # noqa: E501
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

        self._decorate_with = decorate_with

        # NOTE: `_decorated_functions` are not copied in `__deepcopy__`, but it is not
        # necessarily a problem, worst case scenario is a retracing. However, I think
        # there would be no problem copying `_decorated_functions` as long as the
        # decorated functions do not depend on local variables outside of the function.
        # Not doing this, retracing may occur when providing an initial state in
        # `Program.execute_instructions`. Solving this might be better to avoid using
        # `deepcopy` and implement `copy` functions for each `State` subclass.
        self._decorated_functions = dict()

        self._decorator_provided = self._decorate_with is not None

        if self._decorator_provided:
            self.decorator = self._decorator

        self.sqrtm = tf.linalg.sqrtm

        self.range = tf.range

    @property
    def _custom_gradient_enabled(self):
        return not self._decorator_provided and self._tf.executing_eagerly()

    @property
    def forward_pass_np(self):
        return self.fallback_np if self._custom_gradient_enabled else self.np

    @property
    def custom_gradient(self):
        return (
            self._tf.custom_gradient
            if self._custom_gradient_enabled
            else super().custom_gradient
        )

    def _decorator(self, func):
        if func.__name__ in self._decorated_functions:
            return self._decorated_functions[func.__name__]

        decorated = self._decorate_with(func)

        self._decorated_functions[func.__name__] = decorated

        return decorated

    def preprocess_input_for_custom_gradient(self, value):
        if not self._custom_gradient_enabled:
            return value

        return value.numpy() if self._tf.is_tensor(value) else value

    def block_diag(self, *arrs):
        block_diagonalized = self._tf.linalg.LinearOperatorBlockDiag(
            [self._tf.linalg.LinearOperatorFullMatrix(arr) for arr in arrs]
        )

        return block_diagonalized.to_dense()

    def assign(self, array, index, value):
        # NOTE: This method is very limited, and is a bit hacky, since TF does not
        # support item assignment through its NumPy API.

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

    def gather_along_axis_1(self, array, indices):
        # NOTE: Gather along axis 1 was terribly slow in Tensorflow, see
        # https://github.com/tensorflow/ranking/issues/160.

        np = self.fallback_np

        size = array.shape[0]

        size_range = np.arange(size)

        reshaped_indices = []

        for row in size_range:
            reshaped_indices.append(
                np.stack([np.full(indices.shape, row), indices], axis=2)
            )

        return self._tf.gather_nd(array, np.array(reshaped_indices))

    def transpose(self, matrix):
        # NOTE: Similarly to `tf.gather(..., axis=1)`, `tf.transpose` is also pretty
        # slow when JIT compiled, and its einsum implementation is somehow faster.

        return self.np.einsum("ij->ji", matrix)

    def accumulator(self, dtype, size, **kwargs):
        if self._custom_gradient_enabled:
            return super().accumulator(dtype, size, **kwargs)

        return self._tf.TensorArray(dtype=dtype, size=size, **kwargs)

    def write_to_accumulator(self, accumulator, index, value):
        if self._custom_gradient_enabled:
            return super().write_to_accumulator(accumulator, index, value)

        return accumulator.write(index, value)

    def stack_accumulator(self, accumulator):
        if self._custom_gradient_enabled:
            return super().stack_accumulator(accumulator)

        return accumulator.stack()

    def __tf_tracing_type__(self, context):
        # NOTE: We need to create a `TraceType` for `TensorflowConnector` to avoid
        # retracing, but it cannot be defined on module level due to the dependence
        # on `tensorflow`.
        # See `https://www.tensorflow.org/guide/function#use_the_tracing_protocol`_.

        class _TrivialTraceType(self._tf.types.experimental.TraceType):
            def __init__(self, connector):
                self.connector = connector

            def is_subtype_of(self, other):
                return True

            def most_specific_common_supertype(self, others):
                return self

            def placeholder_value(self, placeholder_context=None):
                return self.connector

            def __eq__(self, other):
                return True

            def __hash__(self):
                return 1

        return _TrivialTraceType(self)

    def permanent(self, matrix, rows, columns):
        raise NotImplementedError()

    def hafnian(self, matrix, reduce_on):
        raise NotImplementedError()

    def loop_hafnian(self, matrix, diagonal, reduce_on):
        raise NotImplementedError()

    def loop_hafnian_batch(self, matrix, diagonal, reduce_on, cutoff):
        raise NotImplementedError()
