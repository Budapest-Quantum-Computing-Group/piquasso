#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np


def is_unitary(matrix, tol=1e-10):
    """
    Args:
        tol (float, optional): The tolerance for testing the unitarity.
            Defaults to `1e-10`.

    Returns:
        bool: `True` if the current object is unitary within the specified
            tolerance `tol`, else `False`.
    """
    return (
        matrix @ matrix.conjugate().transpose() - np.identity(matrix.shape[0]) < tol
    ).all()


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.transpose())


def is_selfadjoint(matrix):
    return np.allclose(matrix, matrix.conjugate().transpose())


def direct_sum(*args):
    # TODO: Omit recursions!
    if len(args) == 2:
        a = args[0]
        b = args[1]
        dsum = np.zeros(np.add(a.shape, b.shape), dtype=complex)
        dsum[: a.shape[0], : a.shape[1]] = a
        dsum[a.shape[0]:, a.shape[1]:] = b

        return dsum

    return direct_sum(args[0], direct_sum(*args[1:]))
