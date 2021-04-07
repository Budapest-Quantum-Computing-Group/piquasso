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


def hermite_multidim(B, n, alpha):
    try:
        index = n.index(next(filter(lambda x: x != 0, n)))
    except StopIteration:
        return 1.0

    if sum(n) == 1:
        return np.dot(B[index, :], alpha)

    m = n[index] - 1

    m_prime = np.array(
        [
            value * hermite_multidim(B, m[index] - 1, alpha)
            for index, value
            in enumerate(m)
        ]
    )

    return (
        np.dot(B[index, :], alpha) * hermite_multidim(B, m, alpha)
        - np.dot(B[index, :], m_prime)
    )


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
