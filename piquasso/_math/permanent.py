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

from piquasso._math.combinatorics import comb


def permanent(matrix, rows, cols):
    """Calculates the permanent of a matrix given row and column repetitions.

    Translated from PiquassoBoost.

    Implements Eq. (8) from https://arxiv.org/pdf/2309.07027.pdf.
    """
    mtx2 = 2 * matrix

    delta_limits, minimal_index = _calculate_delta_limits_and_minimal_index(rows)

    if minimal_index == len(rows):
        return 1.0

    col_sum = rows @ matrix

    permanent = _iterate_over_deltas(
        col_sum,
        minimal_index,
        cols,
        mtx2,
        delta_limits,
    )

    sum_multiplicities = sum(rows)

    permanent = permanent / 2 ** (sum_multiplicities - 1)

    return permanent


@nb.njit(cache=True)
def _calculate_delta_limits_and_minimal_index(rows):
    minimal_index = rows.shape[0]

    delta_limits = np.zeros(minimal_index, dtype=rows.dtype)

    for i in range(0, rows.shape[0]):
        if rows[i] > 0:
            if minimal_index > i:
                delta_limits[i] = rows[i] - 1
                minimal_index = i
            else:
                delta_limits[i] = rows[i]
        else:
            delta_limits[i] = 0

    return delta_limits, minimal_index


@nb.njit(parallel=True, cache=True)
def _iterate_over_deltas(
    col_sum,
    index_min,
    cols,
    mtx2,
    delta_limits,
):
    outer_sum = 0.0

    sign = 1
    current_multiplicity = 1

    for idx in nb.prange(index_min, mtx2.shape[0]):
        local_sign = sign

        col_sum_new = np.copy(col_sum)

        inner_sum = 0.0

        for index_of_multiplicity in range(1, delta_limits[idx] + 1):
            col_sum_new -= mtx2[idx]

            local_sign *= -1
            # NOTE: Ideally, the function itself would be called, but Numba does not
            # support recursion with parallelization. As a partial solution, only the
            # outermost for loop is parallelized, and the same function is copied
            # without the parallelization into `_iterate_over_deltas_recursively`, but
            # containing a recursion call.
            inner_sum += _iterate_over_deltas_recursively(
                col_sum=col_sum_new,
                sign=local_sign,
                index_min=idx + 1,
                current_multiplicity=(
                    current_multiplicity
                    * comb(delta_limits[idx], index_of_multiplicity)
                ),
                cols=cols,
                mtx2=mtx2,
                delta_limits=delta_limits,
            )

        outer_sum += inner_sum

    return outer_sum + current_multiplicity * sign * np.prod(col_sum**cols)


@nb.njit(cache=True)
def _iterate_over_deltas_recursively(
    col_sum,
    sign,
    index_min,
    current_multiplicity,
    cols,
    mtx2,
    delta_limits,
):
    # NOTE: This is exactly the same function as `_iterate_over_deltas`, but not
    # parallelized, since Numba does not handle recursion and `parallel=True` well.
    outer_sum = 0.0

    for idx in range(index_min, mtx2.shape[0]):
        local_sign = sign

        col_sum_new = np.copy(col_sum)

        inner_sum = 0.0

        for index_of_multiplicity in range(1, delta_limits[idx] + 1):
            col_sum_new -= mtx2[idx]

            local_sign *= -1
            inner_sum += _iterate_over_deltas_recursively(
                col_sum=col_sum_new,
                sign=local_sign,
                index_min=idx + 1,
                current_multiplicity=(
                    current_multiplicity
                    * comb(delta_limits[idx], index_of_multiplicity)
                ),
                cols=cols,
                mtx2=mtx2,
                delta_limits=delta_limits,
            )

        outer_sum += inner_sum

    return outer_sum + current_multiplicity * sign * np.prod(col_sum**cols)
