#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

import functools

import numpy as np

from scipy.special import comb


def get_operator_index(modes: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note:
        For indexing of numpy arrays, see
        https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
    """

    transformed_columns = np.array([modes] * len(modes))
    transformed_rows = transformed_columns.transpose()

    return transformed_rows, transformed_columns


def get_auxiliary_operator_index(
    modes: Tuple[int, ...], auxiliary_modes: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    auxiliary_rows = tuple(np.array([modes] * len(auxiliary_modes)).transpose())

    return auxiliary_rows, auxiliary_modes


@functools.lru_cache(maxsize=None)
def get_index_in_fock_space(element: Tuple[int, ...]) -> int:
    sum_ = 0
    accumulator = 0
    for i in range(len(element)):
        sum_ += element[-1 - i]
        accumulator += comb(sum_ + i, i + 1, exact=True)

    return accumulator


@functools.lru_cache(maxsize=None)
def get_index_in_fock_subspace(element: Tuple[int, ...]) -> int:
    sum_ = 0
    accumulator = 0
    for i in range(len(element) - 1):
        sum_ += element[-1 - i]
        accumulator += comb(sum_ + i, i + 1, exact=True)

    return accumulator


def get_auxiliary_modes(d: int, modes: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(np.delete(np.arange(d), modes))
