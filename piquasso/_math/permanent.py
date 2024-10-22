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

from numba import int64

from piquasso._math.combinatorics import comb


@nb.njit(cache=True, parallel=True)
def permanent(matrix, rows, cols):
    """Calculates the permanent of a matrix given row and column repetitions.

    Translated from PiquassoBoost, original implementation:
    https://github.com/Budapest-Quantum-Computing-Group/piquassoboost/blob/main/piquassoboost/sampling/source/BBFGPermanentCalculatorRepeated.hpp

    Implements Eq. (8) from https://arxiv.org/pdf/2309.07027.pdf.
    """  # noqa: E501

    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)

    # Determine minimal nonzero element
    min_idx = 0
    minelem = 0
    for i in range(len(rows)):
        if minelem == 0 or rows[i] < minelem and rows[i] != 0:
            minelem = rows[i]
            min_idx = i

    if len(rows) > 0 and minelem != 0:
        rows_ = np.empty(len(rows) + 1, dtype=np.int64)
        rows_[0] = 1
        rows_[1:] = rows
        rows_[1 + min_idx] -= 1

        matrix_ = np.empty(
            shape=(matrix.shape[0] + 1, matrix.shape[1]), dtype=matrix.dtype
        )

        matrix_[0] = matrix[min_idx]

        matrix_[1:] = matrix
        rows = rows_
        matrix = matrix_

    sum_rows = np.sum(rows)
    sum_cols = np.sum(cols)

    if matrix.shape[0] == 0 or matrix.shape[1] == 0 or sum_rows == 0 or sum_cols == 0:
        return 1.0

    if matrix.shape[0] == 1:
        ret = 1.0
        for idx in range(len(cols)):
            for _ in range(cols[idx]):
                ret *= matrix[idx, 0]

        return ret

    mtx2 = matrix * 2

    n_ary_limits = np.empty(len(rows) - 1, dtype=np.int64)

    for idx in range(len(n_ary_limits)):
        n_ary_limits[idx] = rows[idx + 1] + 1

    idx_max = n_ary_limits[0]
    for idx in range(1, len(n_ary_limits)):
        idx_max *= n_ary_limits[idx]

    nthreads = nb.config.NUMBA_NUM_THREADS

    concurrency = min(nthreads, idx_max, 32)

    permanent = 0.0

    for job_idx in nb.prange(concurrency):
        partial_permanent = 0.0

        work_batch = idx_max // concurrency
        initial_offset = job_idx * work_batch
        offset_max = (job_idx + 1) * work_batch - 1
        if job_idx == concurrency - 1:
            offset_max = idx_max - 1

        gcode_counter = NaryGrayCodeCounter(n_ary_limits, initial_offset)

        gcode_counter.offset_max = offset_max
        gcode = gcode_counter.gray_code
        binomial_coeff = 1

        colsum = np.copy(matrix[0])

        minus_signs_all = 0

        row_idx = 1

        for idx in range(len(gcode)):
            minus_signs = gcode[idx]
            rows_current = rows[idx + 1]

            for col_idx in range(len(cols)):
                colsum[col_idx] += matrix[row_idx, col_idx] * (
                    rows_current - 2 * minus_signs
                )

            minus_signs_all += minus_signs

            binomial_coeff *= comb(rows_current, minus_signs)

            row_idx += 1

        parity = 1 if (minus_signs_all % 2 == 0) else -1

        colsum_prod = parity
        for idx in range(len(cols)):
            for _ in range(cols[idx]):
                colsum_prod *= colsum[idx]

        partial_permanent += colsum_prod * binomial_coeff

        for idx in range(initial_offset + 1, offset_max + 1):
            flag, changed_index, value_prev, value = gcode_counter.next()
            if flag:
                break

            parity = -parity

            row_offset = changed_index + 1
            colsum_prod = parity
            for col_idx in range(len(cols)):
                if value_prev < value:
                    colsum[col_idx] -= mtx2[row_offset, col_idx]

                else:
                    colsum[col_idx] += mtx2[row_offset, col_idx]

                for _ in range(cols[col_idx]):
                    colsum_prod *= colsum[col_idx]

            rows_current = rows[changed_index + 1]
            binomial_coeff = (
                (binomial_coeff * value_prev / (rows_current - value))
                if value < value_prev
                else (binomial_coeff * (rows_current - value_prev) / value)
            )

            partial_permanent += colsum_prod * binomial_coeff

        permanent += partial_permanent

    permanent /= 2 ** (sum_rows - 1)

    return permanent


@nb.experimental.jitclass(
    [
        ("gray_code", int64[:]),
        ("n_ary_limits", int64[:]),
        ("counter_chain", int64[:]),
        ("offset_max", int64),
        ("offset", int64),
    ]
)
class NaryGrayCodeCounter(object):
    def __init__(self, n_ary_limits_in, initial_offset):
        self.n_ary_limits = np.copy(n_ary_limits_in)
        if len(self.n_ary_limits) == 0:
            self.offset_max = 0
            self.offset = 0
            return

        self.offset_max = self.n_ary_limits[0]
        for idx in range(1, len(self.n_ary_limits)):
            self.offset_max *= self.n_ary_limits[idx]

        self.offset_max -= 1
        self.offset = initial_offset

        self._initialize(initial_offset)

    def _initialize(self, initial_offset):
        if initial_offset < 0 or initial_offset > self.offset_max:
            raise

        self.counter_chain = np.empty_like(self.n_ary_limits)

        for idx in range(len(self.n_ary_limits)):
            self.counter_chain[idx] = initial_offset % self.n_ary_limits[idx]
            initial_offset /= self.n_ary_limits[idx]

        self.gray_code = np.empty_like(self.n_ary_limits)
        parity = 0
        for jdx in range(len(self.n_ary_limits) - 1, -1, -1):
            self.gray_code[jdx] = (
                self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
                if parity
                else self.counter_chain[jdx]
            )
            parity = parity ^ (self.gray_code[jdx] & 1)

    def next(self):
        changed_index = 0

        if self.offset >= self.offset_max:
            return True, 0, 0, 0

        update_counter = True
        counter_chain_idx = 0
        while update_counter:

            if (
                self.counter_chain[counter_chain_idx]
                < self.n_ary_limits[counter_chain_idx] - 1
            ):
                self.counter_chain[counter_chain_idx] += 1
                update_counter = False

            elif (
                self.counter_chain[counter_chain_idx]
                == self.n_ary_limits[counter_chain_idx] - 1
            ):
                self.counter_chain[counter_chain_idx] = 0
                update_counter = True

            counter_chain_idx += 1

        parity = 0
        for jdx in range(len(self.n_ary_limits) - 1, -1, -1):
            gray_code_new_val = (
                self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
                if parity
                else self.counter_chain[jdx]
            )
            parity = parity ^ (gray_code_new_val & 1)

            if gray_code_new_val != self.gray_code[jdx]:
                value_prev = self.gray_code[jdx]
                value = gray_code_new_val
                self.gray_code[jdx] = gray_code_new_val
                changed_index = jdx
                break

        self.offset += 1

        return False, changed_index, value_prev, value
