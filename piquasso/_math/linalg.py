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
from typing import Iterable, TypeVar, Tuple

import numpy as np
import numpy.typing as npt

from scipy.linalg import block_diag

TNum = TypeVar("TNum", np.float64, np.complex128)


def is_unitary(matrix: npt.NDArray[TNum]) -> bool:
    return np.allclose(
        matrix @ matrix.conjugate().transpose(), np.identity(matrix.shape[0])
    )


def is_symmetric(matrix: npt.NDArray[TNum]) -> bool:
    return np.allclose(matrix, matrix.transpose())


def is_selfadjoint(matrix: npt.NDArray[TNum]) -> bool:
    return np.allclose(matrix, matrix.conjugate().transpose())


def is_positive_semidefinite(matrix: npt.NDArray[TNum]) -> bool:
    eigenvalues = np.linalg.eigvals(matrix)

    return all(
        eigenvalue >= 0.0 or np.isclose(eigenvalue, 0.0)
        for eigenvalue in eigenvalues
    )


def is_square(matrix: npt.NDArray[TNum]) -> bool:
    shape = matrix.shape
    return len(shape) == 2 and shape[0] == shape[1]


def is_symplectic(matrix: npt.NDArray[TNum]) -> bool:
    if not is_square(matrix):
        return False

    d = len(matrix) // 2

    K = block_diag(np.identity(d), -np.identity(d))

    return np.allclose(matrix @ K @ matrix.conj().T, K)


def is_invertible(matrix: npt.NDArray[TNum]) -> bool:
    return (
        is_square(matrix)
        and np.linalg.matrix_rank(matrix) == matrix.shape[0]
    )


def symplectic_form(d: int) -> npt.NDArray[np.intc]:
    one_mode_symplectic_form = np.array([[0, 1], [-1, 0]])

    symplectic_form = block_diag(*([one_mode_symplectic_form] * d))

    return symplectic_form


def reduce_(array: npt.NDArray[TNum], reduce_on: Iterable[int]) -> npt.NDArray[TNum]:
    proper_index = []

    for index, multiplier in enumerate(reduce_on):
        proper_index.extend([index] * multiplier)

    if array.ndim == 1:
        return array[proper_index]

    return array[np.ix_(proper_index, proper_index)]


def block_reduce(
    array: npt.NDArray[TNum], reduce_on: Tuple[int, ...]
) -> npt.NDArray[TNum]:
    return reduce_(array, reduce_on=(reduce_on * 2))
