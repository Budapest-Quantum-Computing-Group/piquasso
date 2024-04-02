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

from piquasso._math.permanent import permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from piquasso.api.calculator import BaseCalculator


class _BuiltinCalculator(BaseCalculator):
    """Base class for built-in calculators."""

    range = range

    def custom_gradient(self, func):
        def noop_grad(*args, **kwargs):
            result, _ = func(*args, **kwargs)
            return result

        return noop_grad

    def accumulator(self, dtype, size, **kwargs):
        return []

    def write_to_accumulator(self, accumulator, index, value):
        accumulator.append(value)

        return accumulator

    def stack_accumulator(self, accumulator):
        return self.forward_pass_np.stack(accumulator)

    def decorator(self, func):
        return func

    def gather_along_axis_1(self, array, indices):
        return array[:, indices]

    def transpose(self, matrix):
        return self.np.transpose(matrix)


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

        self.permanent = permanent
        self.hafnian = hafnian_with_reduction
        self.loop_hafnian = loop_hafnian_with_reduction

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

    Example usage::

        import piquasso as pq
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

    Note:
        Non-deterministic operations like
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement` are
        non-differentiable, please use a deterministic attribute of the resulting state
        instead.

    Note:
        Currently JIT compilation only works with the config variable `normalize=False`.
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
        # NOTE: We need to create a `TraceType` for `TensorflowCalculator` to avoid
        # retracing, but it cannot be defined on module level due to the dependence
        # on `tensorflow`.
        # See `https://www.tensorflow.org/guide/function#use_the_tracing_protocol`_.

        class _TrivialTraceType(self._tf.types.experimental.TraceType):
            def __init__(self, calculator):
                self.calculator = calculator

            def is_subtype_of(self, other):
                return True

            def most_specific_common_supertype(self, others):
                return self

            def placeholder_value(self, placeholder_context=None):
                return self.calculator

            def __eq__(self, other):
                return True

            def __hash__(self):
                return 1

        return _TrivialTraceType(self)


class JaxCalculator(_BuiltinCalculator):
    """The calculations for a simulation using JAX.

    Example usage::

        import numpy as np
        import piquasso as pq
        from jax import jit, grad

        def example_func(r, theta):
            jax_calculator = pq.JaxCalculator()

            simulator = pq.PureFockSimulator(
                d=2,
                config=pq.Config(cutoff=5, dtype=np.float32, normalize=False),
                calculator=jax_calculator,
            )

            with pq.Program() as program:
                pq.Q() | pq.Vacuum()

                pq.Q(0) | pq.Displacement(r=r)

                pq.Q(0, 1) | pq.Beamsplitter(theta)

            state = simulator.execute(program).state

            return state.fock_probabilities[0]

        compiled_func = jit(example_func)

        vacuum_probability = compiled_func(0.1, np.pi / 7)

        compiled_grad_func = jit(grad(compiled_func))

        vacuum_probability_grad = compiled_grad_func(0.2, np.pi / 11)

    Note:
        This feature is still experimental.

    Note:
        Currently JIT compilation only works with the config variable `normalize=False`.

    Note:
        Only CPU calculations are supported currently.
    """

    def __init__(self):
        try:
            import jax.numpy as jnp
            import jax
        except ImportError:
            raise ImportError(
                "You have invoked a feature which requires 'jax'.\n"
                "You can install JAX via:\n"
                "\n"
                "pip install piquasso[jax]"
            )

        # NOTE: Because lots of calculations are still done using NumPy, and NumPy
        # prefers double precision, Piquasso uses double precision too, therefore it is
        # better to enable this config variable. Theoretically, one could set
        # `pq.Config(dtype=np.float32)`, but this might not always work.
        from jax import config

        config.update("jax_enable_x64", True)

        self.np = jnp
        self._scipy = jax.scipy
        self.fallback_np = np
        self.forward_pass_np = jnp
        self.block_diag = jax.scipy.linalg.block_diag
        self.block = jnp.block
        self.logm = partial(jax.scipy.linalg.funm, func=jnp.log)
        self.expm = jax.scipy.linalg.expm
        self.powm = jnp.linalg.matrix_power
        self.sqrtm = jax.scipy.linalg.sqrtm
        self.svd = jnp.linalg.svd

    def preprocess_input_for_custom_gradient(self, value):
        return value

    def assign(self, array, index, value):
        return array.at[index].set(value)

    def scatter(self, indices, updates, shape):
        embedded_matrix = self.np.zeros(shape, dtype=updates[0].dtype)
        indices_array = self.np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])

        return embedded_matrix.at[composite_index].set(self.np.array(updates))

    def polar(self, a, side):
        # NOTE: The default QDWH algorithm does not support left polar decomposition
        # in `jax.scipy.linalg.polar`, so we have to switch to SVD.
        return self._scipy.linalg.polar(a, side, method="svd")
