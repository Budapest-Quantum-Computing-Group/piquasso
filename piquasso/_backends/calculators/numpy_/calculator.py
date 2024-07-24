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
import numba as nb

from piquasso._math.permanent import permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from ..calculator import BuiltinCalculator


class NumpyCalculator(BuiltinCalculator):
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
        self.svd = np.linalg.svd
        self.schur = scipy.linalg.schur

        self.permanent = permanent
        self.hafnian = hafnian_with_reduction
        self.loop_hafnian = loop_hafnian_with_reduction

    def sqrtm(self, matrix):
        return scipy.linalg.sqrtm(matrix).astype(np.complex128)

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

    def calculate_interferometer_on_fock_space(self, interferometer, helper_indices):
        # breakpoint()
        return _calculate_interferometer_on_fock_space(interferometer, helper_indices)


@nb.njit
def _calculate_interferometer_on_fock_space(interferometer, helper_indices):
    """NOTE:
        index_dict = {
            "subspace_index_tensor": index_tuple[0],
            "first_nonzero_index_tensor": index_tuple[1],
            "first_subspace_index_tensor": index_tuple[2],
            "sqrt_occupation_numbers_tensor": index_tuple[3],
            "sqrt_first_occupation_numbers_tensor": index_tuple[4],
        }
    """
    cutoff = len(helper_indices[0]) + 2
    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)


    for n in range(2, cutoff):
        subspace_indices = helper_indices[0][n - 2]
        first_subspace_indices = helper_indices[2][n - 2]

        first_nonzero_indices = helper_indices[1][n - 2]

        sqrt_occupation_numbers = helper_indices[3][n - 2]
        sqrt_first_occupation_numbers = helper_indices[4][n - 2]

        first_part_partially_indexed = interferometer[first_nonzero_indices]
        first_sub_repr = subspace_representations[n - 1][first_subspace_indices]


        result_shape = (first_sub_repr.shape[0], subspace_indices.shape[0], subspace_indices.shape[1])
        second2 = np.empty(result_shape, dtype=np.complex128)

        for (idx, row) in enumerate(first_sub_repr):
            second2[idx] = np.take(row, subspace_indices)

        result_shape = (first_part_partially_indexed.shape[0], sqrt_occupation_numbers.shape[0])
        """
        matrix = np.einsum(
                "ij,kj,kij->ki",
                sqrt_occupation_numbers,
                first_part_partially_indexed,
                second,
            )

        matrix2 = np.zeros(result_shape, dtype=np.complex128)

        for k in nb.prange(first_part_partially_indexed.shape[0]):
            for i in range(sqrt_occupation_numbers.shape[0]):
                sum_value = 0
                for j in range(sqrt_occupation_numbers.shape[1]):
                    sum_value += sqrt_occupation_numbers[i,j] * first_part_partially_indexed[k,j] * second2[k,i,j]
                matrix2[k, i] += sum_value
        """
        matrix1 = np.zeros(result_shape, dtype=np.complex128)

        for k in range(first_part_partially_indexed.shape[0]):
            for i in range(sqrt_occupation_numbers.shape[0]):
                matrix1[k, i] += np.sum(sqrt_occupation_numbers[i,:] * first_part_partially_indexed[k,:] * second2[k,i,:])

        new_subspace_representation = matrix1 / sqrt_first_occupation_numbers[:, None]

        subspace_representations.append(
            new_subspace_representation.astype(interferometer.dtype)
        )

    return subspace_representations
