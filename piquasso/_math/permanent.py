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

from typing import List, Tuple, Any

from functools import lru_cache

import numpy as np

from piquasso._math.linalg import assym_reduce


@lru_cache()
def _generate_gray_code_update_indices(n: int) -> List[int]:
    """
    Computes and returns the update indices for Gray code iteration.

    :param n:   The length of the code.

    :return:    The list of indices to bit-flip during iteration.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    subproblem_update_indices = _generate_gray_code_update_indices(n - 1)

    return (
        subproblem_update_indices + [n - 1] + list(reversed(subproblem_update_indices))
    )


def glynn_gray_permanent(
    matrix: np.ndarray, rows: Tuple[int, ...], columns: Tuple[int, ...], np: Any
) -> complex:
    n = sum(rows)

    if n == 0:
        return complex(1.0)

    reduced_matrix = assym_reduce(matrix, rows, columns)

    sums = []

    for j in range(n):
        sums.append(sum(reduced_matrix[:, j]))

    permanent = np.prod(sums)

    update_indices = _generate_gray_code_update_indices(n - 1)

    multiplier = 1
    delta = [1] * n

    for i in update_indices:
        multiplier = -multiplier
        delta[i] = -delta[i]

        for j in range(n):
            sums[j] += 2 * delta[i] * reduced_matrix[i][j]

        permanent += multiplier * np.prod(sums)

    return permanent / 2 ** (n - 1)


def np_glynn_gray_permanent(
    matrix: np.ndarray, rows: Tuple[int, ...], columns: Tuple[int, ...]
) -> complex:
    return glynn_gray_permanent(matrix, rows, columns, np)
