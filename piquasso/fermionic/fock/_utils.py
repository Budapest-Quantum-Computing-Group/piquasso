#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

import numba as nb

import numpy as np

from functools import lru_cache

from piquasso._math.indices import get_auxiliary_modes

from .._utils import (
    get_fock_space_index,
    get_cutoff_fock_space_dimension,
    next_second_quantized,
)


@lru_cache()
@nb.njit(cache=True)
def calculate_indices_for_controlled_phase(d, cutoff, modes):
    int_dtype = np.int64

    full_occ_number = np.empty(d, dtype=int_dtype)

    aux_modes = get_auxiliary_modes(d, modes)

    full_occ_number[modes[0]] = 1
    full_occ_number[modes[1]] = 1

    size = get_cutoff_fock_space_dimension(d - 2, cutoff)

    indices = np.empty(size, dtype=int_dtype)

    aux_occ_number = np.zeros(d - 2, dtype=int_dtype)

    for i in range(size):
        full_occ_number[aux_modes] = aux_occ_number

        index = get_fock_space_index(full_occ_number)

        indices[i] = index

        aux_occ_number = next_second_quantized(aux_occ_number)

    return indices


@nb.njit(cache=True)
def calculate_indices_for_ising_XX(d, cutoff, modes):
    int_dtype = np.int64

    full_occ_number = np.empty(d, dtype=int_dtype)

    aux_modes = get_auxiliary_modes(d, modes)

    size = get_cutoff_fock_space_dimension(d - 2, cutoff)

    indices = np.empty((size, 4), dtype=int_dtype)

    aux_occ_number = np.zeros(d - 2, dtype=int_dtype)

    for i in range(size):
        full_occ_number[aux_modes] = aux_occ_number

        for j in range(4):
            full_occ_number[modes[0]] = j // 2
            full_occ_number[modes[1]] = j % 2

            index = get_fock_space_index(full_occ_number)

            indices[i, j] = index

        aux_occ_number = next_second_quantized(aux_occ_number)

    return indices
