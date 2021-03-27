#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import math

import numpy as np

from scipy.linalg import block_diag

from functools import lru_cache, partial
from itertools import combinations_with_replacement

from .combinatorics import powerset


@lru_cache()
def get_partitions(boxes, particles):
    particles = particles - boxes

    if particles == 0:
        return [np.ones(boxes, dtype=int)]

    masks = np.rot90(np.identity(boxes, dtype=int))

    ret = []

    for c in combinations_with_replacement(masks, particles):
        ret.append(sum(c) + np.ones(boxes, dtype=int))

    return ret


@lru_cache()
def get_X(d):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    return block_diag(*([sigma_x] * d))


def fG(polynom_coefficients, degree):
    outer_sum = 0
    for j in range(1, degree + 1):

        inner_sum = 0
        for partition in get_partitions(j, degree):

            product = 1
            for index in partition:
                product *= polynom_coefficients[index - 1]

            inner_sum += product

        outer_sum += inner_sum / math.factorial(j)

    return outer_sum


def _hafnian(A, polynom_function):
    degree = A.shape[0] // 2

    if degree == 0:
        return 1.0

    ret = 0

    for subset in powerset(range(degree)):
        if not subset:
            continue  # continue, since empty set has no contribution

        indices = []

        for index in subset:
            indices.extend([2 * index, 2 * index + 1])

        polynom_coefficients = polynom_function(A, indices, degree)

        result = fG(polynom_coefficients, degree)

        factor = 1 if ( (degree - len(subset)) % 2 == 0) else -1

        ret += factor * result

    return ret


def _get_polynom_coefficients(A, indices, degree):
    X = get_X(len(indices) // 2)

    eigenvalues = np.linalg.eigvals(
        X @ A[np.ix_(indices, indices)]
    )

    ret = []

    for power in range(1, degree + 1):
        powertrace = 0.0
        for eigval in eigenvalues:
            powertrace += np.power(eigval, power)

        ret.append(powertrace / (2.0 * power))

    return ret


def _get_loop_polynom_coefficients(A, indices, degree):
    AZ = A[np.ix_(indices, indices)]

    X = get_X(len(indices) // 2)

    XAZ = X @ AZ

    eigenvalues = np.linalg.eigvals(XAZ)

    v = np.diag(AZ)

    ret = []

    for power in range(1, degree + 1):
        powertrace = 0.0
        for eigval in eigenvalues:
            powertrace += np.power(eigval, power)

        coefficient = (
            powertrace / power
            + (
                v
                @ np.linalg.matrix_power(XAZ, power - 1)
                @ X
                @ v.T
            )
        ) / 2.0

        ret.append(coefficient)

    return ret


hafnian = partial(_hafnian, polynom_function=_get_polynom_coefficients)

loop_hafnian = partial(_hafnian, polynom_function=_get_loop_polynom_coefficients)
