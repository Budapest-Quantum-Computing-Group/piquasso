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

from piquasso.api.connector import BaseConnector


class BuiltinConnector(BaseConnector):
    """Base class for built-in connectors."""

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

    def embed_in_identity(self, matrix, indices, dim):
        embedded_matrix = self.np.identity(dim, dtype=complex)

        embedded_matrix = self.assign(embedded_matrix, indices, matrix)

        return embedded_matrix

    def calculate_interferometer_on_fock_space(self, interferometer, helper_indices):
        """
        This implementation uses Eq. (71) from
        https://quantum-journal.org/papers/q-2020-11-30-366/pdf/
        """

        np = self.forward_pass_np
        subspace_representations = []

        subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
        subspace_representations.append(interferometer)

        cutoff = len(helper_indices[0]) + 2

        for n in range(2, cutoff):
            subspace_indices = helper_indices[0][n - 2]
            first_subspace_indices = helper_indices[2][n - 2]

            first_nonzero_indices = helper_indices[1][n - 2]
            sqrt_occupation_numbers = helper_indices[3][n - 2]
            sqrt_first_occupation_numbers = helper_indices[4][n - 2]

            first_part_partially_indexed = interferometer[first_nonzero_indices]
            second = self.gather_along_axis_1(
                subspace_representations[n - 1][first_subspace_indices],
                indices=subspace_indices,
            )

            matrix = np.einsum(
                "ij,kj,kij->ki",
                sqrt_occupation_numbers,
                first_part_partially_indexed,
                second,
            )

            new_subspace_representation = (
                matrix / sqrt_first_occupation_numbers[:, None]
            )

            subspace_representations.append(
                new_subspace_representation.astype(interferometer.dtype)
            )

        return subspace_representations

    def pfaffian(self, matrix):
        """
        Lazy function for calculating the pfaffian.

        Note:
            There are faster algorithms, but this is fine for now.
        """
        if matrix.shape[0] == 0:
            return 1.0

        if matrix.shape[0] % 2 == 1:
            return 0.0

        np = self.np

        blocks, O = self.schur(matrix)
        a = np.diag(blocks, 1)[::2]

        return np.prod(a) * np.linalg.det(O)

    def real_logm(self, matrix):
        """Calculates the real logarithm of a matrix.

        Note:
            This function does not verify the existence of the real logarithm, and it
            will return with a wrong result if such matrix is provided. Note also, that
            this algorithm can certainly be made more efficient, but this is fine for
            now.
        """

        np = self.np

        eigvals, U = np.linalg.eig(matrix)

        I = np.identity(2)
        J = np.array([[0, 1], [-1, 0]])
        WdJW = 1j * np.array([[1, 0], [0, -1]])

        D = np.zeros_like(matrix, dtype=complex)

        forbidden_indices = []

        for index, eigval in enumerate(eigvals):
            if index in forbidden_indices:
                continue

            if np.isclose(np.imag(eigval), 0.0) and eigval >= 0.0:
                D = self.assign(D, (index, index), np.log(eigval))

            else:
                for conjugate_index, conjugate_eigval in enumerate(
                    eigvals[index + 1 :]
                ):
                    if np.isclose(eigval.conj(), conjugate_eigval):
                        break

                forbidden_indices.append(index + conjugate_index + 1)

                indices = np.ix_(*[np.array([index, index + conjugate_index + 1])] * 2)

                r = np.abs(eigval)
                phi = np.angle(eigval)

                if np.isclose(np.imag(eigval), 0.0):
                    D = self.assign(D, indices, np.log(r) * I + phi * J)
                else:
                    D = self.assign(D, indices, np.log(r) * I + phi * WdJW)

        return U @ D @ np.linalg.inv(U)
