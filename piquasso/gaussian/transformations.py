#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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

        T \hat{Y} = T (x_1, \dots, x_d, p_1, \dots, p_d)^T
            = (x_1, p_1, \dots, x_d, p_d)^T,

    which is very helpful in :mod:`piquasso.gaussian.state`.

    Args:
        d (int): The number of modes.

    Returns:
        np.array: The basis changing matrix.
    """

    T = np.zeros((2 * d, 2 * d))
    for i in range(d):
        T[2 * i, i] = 1
        T[2 * i + 1, i + d] = 1

    return T
