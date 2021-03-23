#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import math

import numpy as np

from scipy.linalg import block_diag

import functools

from itertools import chain, combinations, combinations_with_replacement


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


@functools.lru_cache()
def get_partitions(boxes, particles):
    particles = particles - boxes

    if particles == 0:
        return [np.ones(boxes, dtype=int)]

    masks = np.rot90(np.identity(boxes, dtype=int))

    ret = []

    for c in combinations_with_replacement(masks, particles):
        ret.append(sum(c) + np.ones(boxes, dtype=int))

    return ret


@functools.lru_cache()
def X(d):
    sigma_x = np.array([[0, 1], [1, 0]])
    return block_diag(*([sigma_x] * d))


def get_polynom_coefficients(matrix, d):
    eigenvalues = np.linalg.eigvals(matrix)

    ret = []

    for power in range(1, d + 1):
        sum_ = 0.0
        for eigval in eigenvalues:
            sum_ += np.power(eigval, power)

        ret.append(sum_ / (2.0 * power))

    return ret


def fG(matrix, d):
    coefficients = get_polynom_coefficients(matrix, d)

    outer_sum = 0
    for j in range(1, d + 1):

        inner_sum = 0
        for partition in get_partitions(j, d):

            product = 1
            for index in partition:
                product *= coefficients[index - 1]

            inner_sum += product

        outer_sum += inner_sum / math.factorial(j)

    return outer_sum


def hafnian(A):
    d = A.shape[0] // 2

    if d == 0:
        return 1.0

    A = X(d) @ A

    _hafnian = 0

    for subset in powerset(range(d)):
        indices = []

        for index in subset:
            indices.extend([2 * index, 2 * index + 1])

        AZ = A[np.ix_(indices, indices)]

        result = fG(AZ, d)

        factor = 1 if ( (d - len(subset)) % 2 == 0) else -1

        _hafnian += factor * result

    return _hafnian
