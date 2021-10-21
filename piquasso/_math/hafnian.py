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

import math
from functools import lru_cache, partial
from itertools import combinations_with_replacement
from typing import List, Callable

import numpy as np
from scipy.linalg import block_diag

from .combinatorics import powerset


@lru_cache()
def get_partitions(boxes: int, particles: int) -> List[np.ndarray]:
    particles = particles - boxes

    if particles == 0:
        return [np.ones(boxes, dtype=int)]

    masks = np.rot90(np.identity(boxes, dtype=int))

    return [
        sum(c) + np.ones(boxes, dtype=int)
        for c in combinations_with_replacement(masks, particles)
    ]


@lru_cache()
def get_X(d: int) -> np.ndarray:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    return block_diag(*([sigma_x] * d))


def fG(polynom_coefficients: List[float], degree: int) -> float:
    outer_sum = 0.0
    for j in range(1, degree + 1):

        inner_sum = 0.0
        for partition in get_partitions(j, degree):

            product = 1.0
            for index in partition:
                product *= polynom_coefficients[index - 1]

            inner_sum += product

        outer_sum += inner_sum / math.factorial(j)

    return outer_sum


def _hafnian(
    A: np.ndarray, polynom_function: Callable[[np.ndarray, List[int], int], List[float]]
) -> float:
    """
    NOTE: If the input matrix `A` has an odd dimension, e.g. 7x7, then the matrix
    should be padded to an even dimension, to e.g. 8x8.
    """
    if len(A) % 2 == 1:
        A = np.pad(A, pad_width=((1, 0), (1, 0)))
        A[0, 0] = 1.0

    degree = A.shape[0] // 2

    if degree == 0:
        return 1.0

    ret = 0.0

    for subset in powerset(range(degree)):
        if not subset:
            continue  # continue, since empty set has no contribution

        indices = []

        for index in subset:
            indices.extend([2 * index, 2 * index + 1])

        polynom_coefficients = polynom_function(A, indices, degree)

        result = fG(polynom_coefficients, degree)

        factor = 1 if ((degree - len(subset)) % 2 == 0) else -1

        ret += factor * result

    return ret


def _get_polynom_coefficients(
    A: np.ndarray, indices: List[int], degree: int
) -> List[float]:
    X = get_X(len(indices) // 2)

    eigenvalues = np.linalg.eigvals(X @ A[np.ix_(indices, indices)])

    ret = []

    for power in range(1, degree + 1):
        powertrace = 0.0
        for eigval in eigenvalues:
            powertrace += np.power(eigval, power)

        ret.append(powertrace / (2.0 * power))

    return ret


def _get_loop_polynom_coefficients(
    A: np.ndarray, indices: List[int], degree: int
) -> List[float]:
    AZ = A[np.ix_(indices, indices)]

    X = get_X(len(indices) // 2)

    XAZ = X @ AZ

    eigenvalues = np.linalg.eigvals(XAZ)

    v = np.diag(AZ)

    ret = []

    for power in range(1, degree + 1):
        powertrace = 0.0
        for eigval in eigenvalues:
            powertrace += np.power(eigval, power)

        coefficient = (
            powertrace / power + (v @ np.linalg.matrix_power(XAZ, power - 1) @ X @ v.T)
        ) / 2.0

        ret.append(coefficient)

    return ret


hafnian = partial(_hafnian, polynom_function=_get_polynom_coefficients)

loop_hafnian = partial(_hafnian, polynom_function=_get_loop_polynom_coefficients)
