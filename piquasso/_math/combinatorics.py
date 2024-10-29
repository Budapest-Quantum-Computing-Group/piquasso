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

import numpy as np
import numba as nb


@nb.njit(cache=True)
def arr_comb(n, k):
    n = np.where((n < 0) | (n < k), 0, n)
    prod = np.ones(n.shape, dtype=np.int64)

    for i in range(k):
        prod *= n - i
        prod = prod // (i + 1)

    return prod


@nb.njit(cache=True)
def comb(n, k):
    if n < 0 or k < 0 or n < k:
        return 0

    prod = 1

    for i in range(k):
        prod *= n - i
        prod = prod // (i + 1)

    return prod


@nb.njit(cache=True)
def sort_and_get_parity(array):
    n = len(array)
    parity = 1
    for n in range(n - 1, 0, -1):
        for i in range(n):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                parity *= -1

    return array, parity


@nb.njit(cache=True)
def partitions(boxes, particles, out=None):
    r"""
    Returns all the possible ways to put a specified number of particles in a specified
    number of boxes in anti-lexicographic order.

    Args:
        boxes: Number of boxes.
        particles: Number of particles.
        out: Optional output array.
    """

    positions = particles + boxes - 1

    if positions == -1 or boxes == 0:
        return np.empty((1, 0), dtype=np.int32)

    size = comb(positions, boxes - 1)

    if out is None:
        result = np.empty((size, boxes), dtype=np.int32)
    else:
        result = out

    separators = np.arange(boxes - 1, dtype=np.int32)
    index = size - 1

    while True:
        prev = -1
        for i in range(boxes - 1):
            result[index, i] = separators[i] - prev - 1
            prev = separators[i]

        result[index, boxes - 1] = positions - prev - 1
        index -= 1

        if index < 0:
            break

        i = boxes - 2
        while separators[i] == positions - (boxes - 1 - i):
            i -= 1

        separators[i] += 1
        for j in range(i + 1, boxes - 1):
            separators[j] = separators[j - 1] + 1

    return result
