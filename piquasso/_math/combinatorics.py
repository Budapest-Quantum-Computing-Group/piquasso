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

from itertools import chain, combinations
from typing import Tuple, Iterable, Iterator, TypeVar

import numpy as np
import numba as nb


@nb.njit
def arr_comb(n, k):
    n = np.where((n < 0) | (n < k), 0, n)
    prod = np.ones(n.shape, dtype=np.int64)

    for i in range(k):
        prod *= n - i
        prod = prod // (i + 1)

    return prod


@nb.njit(cache=True)
def comb(n, k):
    if n < 0 or n < k:
        return 0

    prod = 1

    for i in range(k):
        prod *= n - i
        prod = prod // (i + 1)

    return prod


@nb.njit
def nb_combinations(arr, r):
    n = arr.shape[0]
    indices = np.arange(r)
    result_size = comb(n, r)
    result = np.empty((result_size, r), dtype=arr.dtype)

    def advance(indices, n, r):
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                break
        else:
            return False

        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1

        return True

    result[0, :] = arr[indices]
    k = 1

    while advance(indices, n, r):
        result[k, :] = arr[indices]
        k += 1

    return result


_T = TypeVar("_T")


def powerset(iterable: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(iterable, r) for r in range(len(s) + 1))


@nb.njit
def partitions(boxes, particles):
    size = boxes + particles - 1

    if size == -1 or boxes == 0:
        return np.empty((1, 0), dtype=np.int32)

    index_matrix = nb_combinations(
        np.array(list(range(size)), dtype=np.int32), boxes - 1
    )
    index_matrix = np.flipud(index_matrix)

    starts = np.concatenate(
        (
            np.zeros(shape=(index_matrix.shape[0], 1), dtype=np.int32),
            np.add(index_matrix, 1),
        ),
        axis=1,
    )

    stops = np.concatenate(
        (
            index_matrix,
            np.full(shape=(index_matrix.shape[0], 1), fill_value=size, dtype=np.int32),
        ),
        axis=1,
    )

    return (stops - starts).astype(np.int32)
