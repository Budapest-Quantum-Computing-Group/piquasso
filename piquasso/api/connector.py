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

import abc

from typing import Any, Tuple

import numpy


class BaseConnector(abc.ABC):
    """The base class for encapsulating a framework.

    This class makes dependency injection into Piquasso possible. This way, Piquasso
    calculations can be made independent of the underlying framework (NumPy, TensorFlow,
    JAX) using the framework's implementation of the NumPy API.

    :ivar np: The implemented NumPy API.
    :ivar fallback_np:
        Usually the original NumPy API for calculations that are not required to be
        present on the computation graph.
    :ivar forward_pass_np: NumPy API used in the forward pass.

    See:
        - `Interoperability with NumPy <https://numpy.org/devdocs/user/basics.interoperability.html>`_

    Note:
        Every attribute of this class should be stateless!
    """  # noqa: E501

    allow_conditionals = True

    np: Any
    fallback_np: Any
    forward_pass_np: Any
    range: Any
    sqrtm: Any
    schur: Any

    def __deepcopy__(self, memo: Any) -> "BaseConnector":
        """
        This method exists, because `copy.deepcopy` could not copy the entire modules
        and functions, and we don't need to, since every attribute of this class should
        be stateless.
        """

        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def preprocess_input_for_custom_gradient(self, value):
        """
        Applies modifications to inputs in custom gradients.
        """

    @abc.abstractmethod
    def permanent(
        self, matrix: numpy.ndarray, rows: Tuple[int, ...], columns: Tuple[int, ...]
    ) -> float:
        """Calculates the permanent of a matrix with row and column repetitions."""

    @abc.abstractmethod
    def hafnian(self, matrix: numpy.ndarray, reduce_on: numpy.ndarray) -> float:
        r"""Calculates the hafnian of a matrix with prescribed reduction array.

        This function first performs a reduction by a reduction array :math:`S`, and
        then calculates the hafnian. Succintly, this function should implement
        :math:`A \mapsto \operatorname{haf}(A_{(S)})`.

        Args:
            matrix (numpy.ndarray): The input matrix.
            reduce_on (numpy.ndarray): The reduction array.

        Returns:
            float: The hafnian of the matrix.
        """

    @abc.abstractmethod
    def loop_hafnian(
        self, matrix: numpy.ndarray, diagonal: numpy.ndarray, reduce_on: numpy.ndarray
    ) -> float:
        r"""Calculates the hafnian of a matrix with prescribed reduction array.

        This function first fills the diagonals with :math:`D`, then performs a
        reduction by a reduction array :math:`S` and then calculates the hafnian.
        Succintly, this function should implement
        :math:`A \mapsto \operatorname{lhaf}(\operatorname{filldiag}(A, D)_{(S)})`.

        Args:
            matrix (numpy.ndarray): The input matrix.
            diagonal (numpy.ndarray): The vector which will fill the diagonal.
            reduce_on (numpy.ndarray): The reduction array.

        Returns:
            float: The hafnian of the matrix.
        """

    @abc.abstractmethod
    def loop_hafnian_batch(
        self,
        matrix: numpy.ndarray,
        diagonal: numpy.ndarray,
        reduce_on: numpy.ndarray,
        cutoff: int,
    ) -> float:
        r"""Batch loop hafnian calculation.

        Same as :meth:`loop_hafnian`, but with batching, according to
        https://arxiv.org/abs/2108.01622.
        """

    @abc.abstractmethod
    def assign(self, array, index, value):
        """Item assignment."""

    @abc.abstractmethod
    def scatter(self, indices, updates, shape):
        """Filling an array of a given shape with the given indices and update values.

        Equivalent to :func:`tf.scatter_nd`.
        """

    @abc.abstractmethod
    def embed_in_identity(self, matrix, indices, dim):
        """Embeds a matrix in identity."""

    @abc.abstractmethod
    def block(self, arrays):
        """Assembling submatrices into a single matrix.

        Equivalent to :func:`numpy.block`.
        """

    @abc.abstractmethod
    def block_diag(self, *arrs):
        """Putting together matrices as a block diagonal matrix.

        Equivalent to :func:`scipy.linalg.block_diag`.
        """

    @abc.abstractmethod
    def polar(self, matrix, side="right"):
        """Polar decomposition.

        Equivalent to :func:`scipy.linalg.polar`.

        Args:
            matrix (numpy.ndarray): The input matrix
            side (str, optional): The order of the decomposition. Defaults to "right".
        """

    @abc.abstractmethod
    def logm(self, matrix):
        """Matrix logarithm.

        Equivalent to :func:`scipy.linalg.logm`.

        Args:
            matrix (numpy.ndarray): The input matrix.
        """

    @abc.abstractmethod
    def expm(self, matrix):
        """Matrix exponential.

        Equivalent to :func:`scipy.linalg.expm`.

        Args:
            matrix (numpy.ndarray): The input matrix.
        """

    @abc.abstractmethod
    def powm(self, matrix, power):
        """Matrix power.

        Equivalent to :func:`numpy.linalg.matrix_power`.

        Args:
            matrix (numpy.ndarray): The input matrix.
        """

    @abc.abstractmethod
    def custom_gradient(self, func):
        """Custom gradient wrapper.

        Args:
            func: The function for which custom gradient is defined.
        """

    @abc.abstractmethod
    def accumulator(self, dtype, size, **kwargs):
        """Datatype to collect NumPy arrays.

        Common generalization of a Python list and
        `tf.TensorArray <https://www.tensorflow.org/api_docs/python/tf/TensorArray>`_.
        """

    @abc.abstractmethod
    def write_to_accumulator(self, accumulator, index, value):
        """Append an element to the accumulator.

        Common generalization of a Python list appending and
        `tf.TensorArray.write <https://www.tensorflow.org/api_docs/python/tf/TensorArray#write>`_.
        """  # noqa: E501

    @abc.abstractmethod
    def stack_accumulator(self, accumulator):
        """Stack elements in the accumulator.

        Common generalization of :func:`numpy.stack` and
        `tf.TensorArray.stack <https://www.tensorflow.org/api_docs/python/tf/TensorArray#stack>`_.
        """  # noqa: E501

    @abc.abstractmethod
    def decorator(self, func):
        """Decorates heavy computations in Piquasso.

        Args:
            func: Function to decorate.
        """

    @abc.abstractmethod
    def gather_along_axis_1(self, array, indices):
        """Gathering values along axis 1 of a matrix.

        Note:
            Gather along axis 1 was terribly slow in Tensorflow, see
            https://github.com/tensorflow/ranking/issues/160.
        """

    @abc.abstractmethod
    def transpose(self, matrix):
        """Matrix transposition.

        Args:
            matrix (numpy.ndarray): The input matrix.
        """

    @abc.abstractmethod
    def calculate_interferometer_on_fock_space(self, interferometer, helper_indices):
        """Calculates the interferometer unitary matrix on the Fock space.

        Args:
            interferometer: One-particle unitary corresponding to the interferometer.
            helper_indices: Separately calculated helper indices.

        Returns:
            All the n-particle unitary matrices corresponding to the interferometer up
            to cutoff.
        """

    @abc.abstractmethod
    def pfaffian(self, matrix):
        """Calculates the pfaffian of a matrix."""

    @abc.abstractmethod
    def real_logm(self, matrix):
        """Calculates the real logarithm of a matrix."""
