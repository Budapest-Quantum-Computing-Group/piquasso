#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from ..connector import BuiltinConnector


class TorchConnector(BuiltinConnector):
    """TODO(TR)"""

    def __init__(self):
        try:
            import torch

            from .np_mock import MockNumpy
        except ImportError:
            raise ImportError(
                "You have invoked a feature which requires 'torch'.\n"
                "You can install PyTorch via:\n"
                "\n"
                "pip install piquasso[torch]"
            )

        self.np = self.forward_pass_np = MockNumpy()
        self.torch = torch
        self.fallback_np = np  # NOTE(TR): I assume this is the "last resort" numpy.

    def block_diag(self, arrs):
        return self.torch.block_diag(arrs)

    def block(self, tensors):
        # Horizontal concatenation
        rows = [self.block(row) for row in tensors]
        # Vertical concatenation
        return self.torch.cat(rows, dim=0)

    def _funm(self, matrix, func):
        """Helper function for the ``self.logm`` and ``self.expm`` implementation.

        NOTE: This is the same strategy as for the TensorflowConnector."""
        eigenvalues, U = self.torch.linalg.eig(matrix)

        f_eigenvalues = func(eigenvalues)

        return U @ self.torch.diag(f_eigenvalues) @ self.torch.linalg.inv(U)

    def logm(self, matrix):
        return self._funm(matrix, self.torch.log)

    def expm(self, matrix):
        return self._funm(matrix, self.torch.exp)

    def powm(self, matrix, power):
        return self.torch.linalg.matrix_power(matrix, power)

    def schur(self, matrix):
        # NOTE: Based on the TensorflowConnector implementation.
        _, vecs = self.torch.linalg.eig(matrix)
        Q, _ = self.torch.linalg.qr(vecs)
        D = self.torch.adjoint(Q) @ matrix @ Q
        return D, Q

    def sqrtm(self, A):
        # TODO(TR): AI-generated! Check if that's correct!
        # NOTE: Supposedly mimics TF implementation, and could be slow due
        # to the lack of C++ support.
        T, Z = self.schur(A)

        n = T.shape[-1]
        R = self.torch.zeros_like(T)

        for j in range(n):
            R[j, j] = self.torch.sqrt(T[j, j])
            for i in range(j - 1, -1, -1):
                s = self.torch.dot(R[i, i + 1 : j], R[i + 1 : j, j])
                R[i, j] = (T[i, j] - s) / (R[i, i] + R[j, j])

        res = Z @ R @ Z.mhr()
        return res

    def polar(self, matrix, side="right"):
        # NOTE: Based on the TensorflowConnector implementation.
        P = self.sqrtm(self.torch.conj(matrix) @ matrix.t)
        Pinv = self.torch.linalg.inv(P)

        if side == "right":
            U = matrix @ Pinv
        elif side == "left":
            U = Pinv @ matrix

        return U, P

    def svd(self, matrix):
        # NOTE: torch and numpy have matching return order
        return self.torch.linalg.svd(matrix)

    def scatter(self, indices, updates, shape):
        # NOTE: Based on the JaxConnector implementation
        embedded_matrix = self.torch.zeros(shape, dtype=updates[0].dtype)
        indices_array = self.np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])
        embedded_matrix[composite_index] = self.np.array(updates)
        return embedded_matrix

    def preprocess_input_for_custom_gradient(self, value):
        return value

    # NOTE: These two are not implemented in the
    # ``TensorflowConnector``, so I'm also leaving them here.
    # calculate_interferometer_on_fock_space
    # calculate_interferometer_on_fermionic_fock_space

    def permanent(self, matrix, rows, cols):
        # NOTE: Same as in TensorflowConnector implementation.
        raise NotImplementedError()

    def permanent_laplace(self, matrix, rows, cols):
        # NOTE: Same as in TensorflowConnector implementation.
        raise NotImplementedError()

    def hafnian(self, matrix, reduce_on):
        # NOTE: Same as in TensorflowConnector implementation.
        raise NotImplementedError()

    def loop_hafnian(self, matrix, diagonal, reduce_on):
        # NOTE: Same as in TensorflowConnector implementation.
        raise NotImplementedError()

    def loop_hafnian_batch(self, matrix, diagonal, reduce_on, cutoff):
        # NOTE: Same as in TensorflowConnector implementation.
        raise NotImplementedError()

    def assign(self, array, index, value):
        array[index] = value

        return array
