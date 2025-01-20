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

import numpy as np

import numba as nb

from piquasso.fermionic._utils import (
    get_fock_subspace_dimension,
    get_fock_subspace_index_first_quantized,
    next_first_quantized,
)


@nb.njit(cache=True)
def precalculate_fermionic_passive_linear_indices(n, d):
    dim = get_fock_subspace_dimension(d, n)

    laplace_indices = np.empty((dim, n), dtype=np.int64)
    deleted_indices = np.empty((dim, n), dtype=np.int64)

    first_quantized = np.arange(n)
    for row_idx in range(dim):
        for laplace_index in range(n):
            deleted = np.delete(first_quantized, laplace_index)
            deleted_idx = get_fock_subspace_index_first_quantized(deleted, d)
            deleted_indices[row_idx, laplace_index] = deleted_idx
            laplace_indices[row_idx, laplace_index] = first_quantized[laplace_index]

        first_quantized = next_first_quantized(first_quantized, d)

    return (laplace_indices, deleted_indices)
