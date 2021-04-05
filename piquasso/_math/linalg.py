#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from scipy.linalg import block_diag


def is_unitary(matrix):
    """
    Args:
        tol (float, optional): The tolerance for testing the unitarity.
            Defaults to `1e-10`.

    Returns:
        bool: `True` if the current object is unitary within the specified
            tolerance `tol`, else `False`.
    """
    return np.allclose(
        matrix @ matrix.conjugate().transpose(), np.identity(matrix.shape[0])
    )


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.transpose())


def is_selfadjoint(matrix):
    return np.allclose(matrix, matrix.conjugate().transpose())


def is_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)


def is_square(matrix):
    shape = matrix.shape
    return len(shape) == 2 and shape[0] == shape[1]


def symplectic_form(d):
    one_mode_symplectic_form = np.array([[0, 1], [-1, 0]])

    symplectic_form = block_diag(*([one_mode_symplectic_form] * d))

    return symplectic_form


def block_reduce(array, reduce_on):
    reduce_on *= 2

    proper_index = []

    for index, multiplier in enumerate(reduce_on):
        proper_index.extend([index] * multiplier)

    if array.ndim == 1:
        return array[proper_index]

    return array[np.ix_(proper_index, proper_index)]
