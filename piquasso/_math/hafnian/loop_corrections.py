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

import numba as nb
import numpy as np


@nb.njit(cache=True)
def calculate_loop_corrections(cx_diag_elements, diag_elements, AZ, num_of_modes):
    """
    Calculates loop corrections.
    """
    loop_correction = np.zeros(num_of_modes, dtype=AZ.dtype)

    max_idx = len(cx_diag_elements)

    tmp_vec = np.zeros(max_idx, dtype=AZ.dtype)

    for idx in range(0, num_of_modes):
        tmp = 0.0

        for jdx in range(0, max_idx):
            tmp = tmp + diag_elements[jdx] * cx_diag_elements[jdx]

        loop_correction[idx] = tmp

        tmp = 0.0
        for kdx in range(0, max_idx):
            tmp += AZ[0, kdx] * cx_diag_elements[kdx]

        tmp_vec[0] = tmp

        for jdx in range(1, max_idx):
            tmp = 0.0
            for kdx in range(jdx - 1, max_idx):
                tmp += AZ[jdx, kdx] * cx_diag_elements[kdx]

            tmp_vec[jdx] = tmp

        cx_diag_elements = np.copy(tmp_vec)

    return loop_correction
