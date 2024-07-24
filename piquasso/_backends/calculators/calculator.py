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

from piquasso.api.calculator import BaseCalculator


class BuiltinCalculator(BaseCalculator):
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

    def embed_in_identity(self, matrix, indices, dim):
        embedded_matrix = self.np.identity(dim, dtype=complex)

        embedded_matrix = self.assign(embedded_matrix, indices, matrix)

        return embedded_matrix

    def calculate_interferometer_on_fock_space(self, interferometer, helper_indices):
        np = self.forward_pass_np
        subspace_representations = []

        subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
        subspace_representations.append(interferometer)

        cutoff = len(helper_indices["subspace_index_tensor"]) + 2

        for n in range(2, cutoff):
            subspace_indices = helper_indices["subspace_index_tensor"][n - 2]
            first_subspace_indices = helper_indices["first_subspace_index_tensor"][
                n - 2
            ]

            first_nonzero_indices = helper_indices["first_nonzero_index_tensor"][n - 2]

            sqrt_occupation_numbers = helper_indices["sqrt_occupation_numbers_tensor"][
                n - 2
            ]
            sqrt_first_occupation_numbers = helper_indices[
                "sqrt_first_occupation_numbers_tensor"
            ][n - 2]

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
