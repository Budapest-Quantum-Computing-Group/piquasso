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

from typing import Collection

import numpy as np

from piquasso._math.linalg import reduce_
from piquasso._math.hafnian import loop_hafnian


def modified_hermite_multidim(
    B: np.ndarray, n: Collection[int], alpha: np.ndarray
) -> complex:
    r"""
    For the calculation of the modified Hermite polynomials :math:`G` the following
    equation is used:

    .. math::
        G_{\vec{n}}^{B}(\alpha) = \operatorname{lhaf}(
            \operatorname{filldiag}(B_{\vec{n}}, \alpha_{\vec{n}})
        )
    """

    loop_hafnian_input = -reduce_(B, n)

    np.fill_diagonal(loop_hafnian_input, reduce_(alpha, n))

    return loop_hafnian(loop_hafnian_input)
