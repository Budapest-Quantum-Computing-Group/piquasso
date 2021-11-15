#
# Copyright 2021 Budapest Quantum Computing Group
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

from typing import Iterable, Tuple

import numpy as np

from .validations import all_real_and_positive


def is_unitary(matrix: np.ndarray) -> bool:
    return np.allclose(
        matrix @ matrix.conjugate().transpose(), np.identity(matrix.shape[0])
    )


def is_real(matrix: np.ndarray) -> bool:
    return bool(np.all(np.isreal(matrix)))


def is_2n_by_2n(matrix: np.ndarray) -> bool:
    dim = len(matrix)

    return matrix.shape == (dim, dim) and dim % 2 == 0


def is_real_2n_by_2n(matrix: np.ndarray) -> bool:
    return is_real(matrix) and is_2n_by_2n(matrix)


def is_symmetric(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, matrix.transpose())


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


def reduce_(array: np.ndarray, reduce_on: Iterable[int]) -> np.ndarray:
    proper_index = []

    for index, multiplier in enumerate(reduce_on):
        proper_index.extend([index] * multiplier)

    if array.ndim == 1:
        return array[proper_index]

    return array[np.ix_(proper_index, proper_index)]


def block_reduce(array: np.ndarray, reduce_on: Tuple[int, ...]) -> np.ndarray:
    return reduce_(array, reduce_on=(reduce_on * 2))
