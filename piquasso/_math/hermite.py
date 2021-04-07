#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from scipy.special import factorial


def hermite_kampe(n: int, x: complex, y: complex):
    sum_ = 0.0

    for r in range(n // 2 + 1):
        sum_ += (
            np.power(x, n - 2 * r)
            * np.power(y, r)
        ) / (factorial(n - 2 * r) * factorial(r))

    return factorial(n) * sum_


def hermite_kampe_2dim(
    *,
    n: int, m: int,
    x: complex, y: complex,
    z: complex, u: complex,
    tau: complex
):
    sum_ = 0.0

    for r in range(min(n, m) + 1):
        sum_ += (
            hermite_kampe(m - r, x, y)
            * hermite_kampe(n - r, z, u)
            * np.power(tau, r)
        ) / (
            factorial(m - r)
            * factorial(r)
            * factorial(n - r)
        )

    return factorial(n) * factorial(m) * sum_
