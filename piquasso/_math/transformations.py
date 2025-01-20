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


@nb.njit(cache=True)
def xxpp_to_xpxp_indices(d: int) -> np.ndarray:
    r"""
    Indices for basis changing from the xxpp to the xpxp basis.

    Args:
        d (int): The number of modes.

    Returns:
        numpy.ndarray: The basis changing indices.
    """

    indices = np.empty(2 * d, dtype=nb.int32)

    for i in range(d):
        indices[2 * i] = i
        indices[2 * i + 1] = d + i

    return indices


@nb.njit(cache=True)
def xpxp_to_xxpp_indices(d: int) -> np.ndarray:
    r"""
    Indices for basis changing from the xpxp to the xxpp basis.

    Args:
        d (int): The number of modes.

    Returns:
        numpy.ndarray: The basis changing indices.
    """

    indices = np.empty(2 * d, dtype=nb.int32)

    for i in range(d):
        indices[i] = 2 * i
        indices[d + i] = 2 * i + 1

    return indices
