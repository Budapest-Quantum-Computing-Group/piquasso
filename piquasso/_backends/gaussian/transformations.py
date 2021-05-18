#
# Copyright 2021 Budapest Quantum Computing Group
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

import functools
import numpy as np


@functools.lru_cache()
def quad_transformation(d):
    """
    Basis changing with the basis change operator

    .. math::

        T_{ij} = \delta_{j, 2i-1} + \delta_{j + 2d, 2i}

    Intuitively, it changes the basis as

    .. math::

        T Y = T (x_1, \dots, x_d, p_1, \dots, p_d)^T
            = (x_1, p_1, \dots, x_d, p_d)^T,

    which is very helpful in :mod:`piquasso._backends.gaussian.state`.

    Args:
        d (int): The number of modes.

    Returns:
        numpy.ndarray: The basis changing matrix.
    """

    T = np.zeros((2 * d, 2 * d))
    for i in range(d):
        T[2 * i, i] = 1
        T[2 * i + 1, i + d] = 1

    return T
