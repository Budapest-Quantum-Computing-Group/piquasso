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

import numpy as np

from scipy.linalg import block_diag


def is_unitary(matrix):
    """
    Args:
        tol (float, optional): The tolerance for testing the unitarity.
            Defaults to `1e-10`.

    Returns:
        bool: `True` if the current object is unitary within the specified
            tolerance `tol`, else `False`.
    """
    return np.allclose(
        matrix @ matrix.conjugate().transpose(), np.identity(matrix.shape[0])
    )


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.transpose())


def is_selfadjoint(matrix):
    return np.allclose(matrix, matrix.conjugate().transpose())


def is_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)


def is_square(matrix):
    shape = matrix.shape
    return len(shape) == 2 and shape[0] == shape[1]


def symplectic_form(d):
    one_mode_symplectic_form = np.array([[0, 1], [-1, 0]])

    symplectic_form = block_diag(*([one_mode_symplectic_form] * d))

    return symplectic_form


def block_reduce(array, reduce_on):
    reduce_on *= 2

    proper_index = []

    for index, multiplier in enumerate(reduce_on):
        proper_index.extend([index] * multiplier)

    if array.ndim == 1:
        return array[proper_index]

    return array[np.ix_(proper_index, proper_index)]
