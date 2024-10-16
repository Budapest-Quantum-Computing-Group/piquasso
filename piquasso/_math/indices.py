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
import numba as nb

from piquasso._math.combinatorics import arr_comb, comb


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


@nb.njit(cache=True)
def get_index_in_fock_space(element):
    sum_ = 0
    accumulator = 0
    for i in range(len(element)):
        sum_ += element[-1 - i]
        accumulator += comb(sum_ + i, i + 1)

    return accumulator


@nb.njit(cache=True)
def get_index_in_fock_space_array(basis: np.ndarray) -> np.ndarray:
    sum_ = np.zeros(shape=basis.shape[:-1], dtype=np.int32)
    accumulator = np.zeros(shape=basis.shape[:-1], dtype=np.int32)

    for i in range(basis.shape[-1]):
        sum_ += basis[..., -1 - i]
        accumulator += arr_comb(sum_ + i, i + 1)

    return accumulator


@nb.njit(cache=True)
def get_index_in_fock_subspace(element: np.ndarray) -> int:
    sum_ = 0
    accumulator = 0
    for i in range(len(element) - 1):
        sum_ += element[-1 - i]
        accumulator += comb(sum_ + i, i + 1)

    return accumulator


@nb.njit(cache=True)
def get_index_in_fock_subspace_array(basis: np.ndarray) -> np.ndarray:
    sum_ = np.zeros(shape=basis.shape[:-1], dtype=np.int32)
    accumulator = np.zeros(shape=basis.shape[:-1], dtype=np.int32)

    for i in range(basis.shape[-1] - 1):
        sum_ += basis[..., -1 - i]
        accumulator += arr_comb(sum_ + i, i + 1)

    return accumulator


@nb.njit(cache=True)
def get_auxiliary_modes(d: int, modes: Tuple[int, ...]) -> np.ndarray:
    return np.delete(np.arange(d), modes)
