#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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
