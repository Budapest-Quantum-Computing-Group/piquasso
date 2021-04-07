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

import scipy
import numpy as np


def takagi(matrix, rounding=12):
    """Takagi factorization of complex symmetric matrices.

    Note:

        The singular values have to be rounded due to floating point errors.

        The result is not unique in a sense that different result could be obtained
        by different ordering of the singular values.

    References:
    - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.94.062109
    """

    V, singular_values, W_adjoint = np.linalg.svd(matrix)

    W = W_adjoint.conjugate().transpose()

    singular_value_multiplicity_map = {}

    for index, value in enumerate(singular_values):
        value = np.round(value, decimals=rounding)

        if value not in singular_value_multiplicity_map:
            singular_value_multiplicity_map[value] = [index]
        else:
            singular_value_multiplicity_map[value].append(index)

    diagonal_blocks_for_Q = []

    for indices in singular_value_multiplicity_map.values():
        Z = V[:, indices].transpose() @ W[:, indices]

        diagonal_blocks_for_Q.append(scipy.linalg.sqrtm(Z))

    Q = scipy.linalg.block_diag(*diagonal_blocks_for_Q)

    return singular_values, V @ Q.conjugate()
