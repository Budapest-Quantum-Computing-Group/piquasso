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

from typing import Tuple

import numpy as np

from .validations import all_real_and_positive


def is_unitary(matrix: np.ndarray) -> bool:
    return np.allclose(matrix @ np.conj(matrix).T, np.identity(matrix.shape[0]))


def is_real(matrix: np.ndarray) -> bool:
    return bool(np.all(np.isreal(matrix)))


def is_2n_by_2n(matrix: np.ndarray) -> bool:
    dim = len(matrix)

    return matrix.shape == (dim, dim) and dim % 2 == 0


def is_real_2n_by_2n(matrix: np.ndarray) -> bool:
    return is_real(matrix) and is_2n_by_2n(matrix)


def is_symmetric(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, matrix.transpose())


def is_skew_symmetric(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, -matrix.transpose())


def is_selfadjoint(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, matrix.conjugate().transpose())


def is_positive_semidefinite(matrix: np.ndarray) -> bool:
    eigenvalues = np.linalg.eigvals(matrix)

    return all_real_and_positive(eigenvalues)


def is_square(matrix: np.ndarray) -> bool:
    shape = matrix.shape
    return len(shape) == 2 and shape[0] == shape[1]


def is_invertible(matrix: np.ndarray) -> bool:
    return is_square(matrix) and np.linalg.matrix_rank(matrix) == matrix.shape[0]


def is_diagonal(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, np.diag(np.diag(matrix)))


def reduce_(array, reduce_on):
    particles = np.sum(reduce_on)

    proper_index = np.zeros(particles, dtype=int)

    stride = 0

    for index in range(len(reduce_on)):
        multiplier = reduce_on[index]
        proper_index[stride : stride + multiplier] = index
        stride += multiplier

    if array.ndim == 1:
        return array[proper_index]

    return array[np.ix_(proper_index, proper_index)]


def block_reduce(array: np.ndarray, reduce_on: Tuple[int, ...]) -> np.ndarray:
    return reduce_(array, reduce_on=(reduce_on * 2))


def block_reduce_xpxp(array: np.ndarray, reduce_on: Tuple[int, ...]) -> np.ndarray:
    return reduce_(array, reduce_on=np.repeat(reduce_on, 2))


def assym_reduce(array, row_reduce_on, col_reduce_on):
    particles = np.sum(row_reduce_on)

    proper_row_index = np.zeros(particles, dtype=int)
    proper_col_index = np.zeros(particles, dtype=int)

    row_stride = 0
    col_stride = 0

    for index in range(len(row_reduce_on)):
        row_multiplier = row_reduce_on[index]
        proper_row_index[row_stride : row_stride + row_multiplier] = index
        row_stride += row_multiplier

    for index in range(len(col_reduce_on)):
        col_multiplier = col_reduce_on[index]
        proper_col_index[col_stride : col_stride + col_multiplier] = index
        col_stride += col_multiplier

    return array[np.ix_(proper_row_index, proper_col_index)]


def vector_absolute_square(vector, connector):
    @connector.custom_gradient
    def _vector_absolute_square(v):
        np = connector.forward_pass_np

        absolute_square = np.real((v * np.conj(v)))

        def _vector_absolute_square_grad(upstream):
            return 2 * upstream * v

        return absolute_square, _vector_absolute_square_grad

    return _vector_absolute_square(vector)
