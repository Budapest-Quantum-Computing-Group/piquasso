#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from .combinatorics import powerset


def torontonian(A):
    d = A.shape[0] // 2

    if d == 0:
        return 1.0

    ret = 0.0

    for subset in powerset(range(0, d)):
        index = np.ix_(subset, subset)

        A_reduced = np.block(
            [
                [A[:d, :d][index], A[:d, d:][index]],
                [A[d:, :d][index], A[d:, d:][index]],
            ]
        )

        factor = 1.0 if ( (d - len(subset)) % 2 == 0) else -1.0

        ret += factor / np.sqrt(
            np.linalg.det(
                np.identity(len(A_reduced)) - A_reduced
            ).real + 0.j
        )

    return ret
