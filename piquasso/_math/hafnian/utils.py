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
def _get_inverse_permutation(p):
    s = np.empty(p.size, dtype=np.int32)
    for i in np.arange(p.size):
        s[p[i]] = i

    return s


@nb.njit(cache=True)
def match_occupation_numbers(nvec_orig):
    dtype = nvec_orig.dtype
    nvec = np.copy(nvec_orig)
    edge_reps = []
    edge_indices = []

    if len(nvec_orig) == 1:
        # NOTE: This function only executed when `sum(nvec_orig) % 2 == 0`.
        return np.array([nvec_orig[0] // 2], dtype=dtype), np.array([0, 0], dtype=dtype)

    while sum(nvec) > 1:
        sorter = np.argsort(nvec)
        nvec_sorted = nvec[sorter]

        n0_over_2 = nvec_sorted[-1] // 2
        n1 = nvec_sorted[-2]

        if n0_over_2 > nvec_sorted[-2]:
            nvec_sorted[-1] -= 2 * n0_over_2
            edge_reps.append(n0_over_2)
            edge_indices.extend([sorter[-1], sorter[-1]])

        else:
            nvec_sorted[-1] -= n1
            nvec_sorted[-2] = 0
            edge_reps.append(n1)
            edge_indices.extend([sorter[-1], sorter[-2]])

        sorterinv = _get_inverse_permutation(sorter)
        nvec = nvec_sorted[sorterinv]

    return np.array(edge_reps, dtype=dtype), np.array(edge_indices, dtype=dtype)


@nb.njit(cache=True)
def ix_(matrix, rows, cols):
    sliced_matrix = np.empty(shape=(len(rows), len(cols)), dtype=matrix.dtype)

    for idx in range(len(rows)):
        for jdx in range(len(cols)):
            sliced_matrix[idx, jdx] = matrix[rows[idx], cols[jdx]]

    return sliced_matrix


@nb.njit(cache=True)
def get_kept_edges(edge_reps, index):
    ret = np.empty_like(edge_reps)
    i = 0
    edge_reps_p_1 = edge_reps + 1

    for n in edge_reps_p_1:
        ret[i] = index % n
        index //= n
        i += 1

    return ret
