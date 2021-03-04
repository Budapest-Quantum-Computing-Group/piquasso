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
