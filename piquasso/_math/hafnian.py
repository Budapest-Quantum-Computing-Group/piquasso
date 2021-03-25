#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import math
import random

import numpy as np

from scipy.linalg import block_diag
from scipy.special import factorial

from functools import lru_cache
from itertools import chain, repeat, combinations, combinations_with_replacement

from .linalg import block_reduce, blocks_on_subspace
from ._random import choose_from_cumulated_probabilities


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


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


def generate_gaussian_state_samples(*, husimi, modes, shots, cutoff):
    d = len(modes)

    identity = np.identity(d)
    zeros = np.zeros_like(identity)
    X = np.block(
        [
            [zeros, identity],
            [identity, zeros],
        ],
    )

    A = X @ ((np.identity(2 * d) - np.linalg.inv(husimi)))

    @lru_cache(maxsize=None)
    def get_A_on_subspace(subspace_modes):
        return blocks_on_subspace(
            matrix=A,
            subspace_indices=subspace_modes,
        )

    @lru_cache(maxsize=None)
    def get_husimi_factor_on_subspace(subspace_modes):
        husimi_on_subspace = blocks_on_subspace(
            matrix=husimi,
            subspace_indices=subspace_modes,
        )

        return np.sqrt(np.linalg.det(husimi_on_subspace))

    @lru_cache(maxsize=None)
    def get_probability(*, subspace_modes, occupation_numbers):
        A_on_subspace = get_A_on_subspace(subspace_modes=subspace_modes)
        A_reduced = block_reduce(
            A_on_subspace, reduction_indices=occupation_numbers
        )
        husimi_factor = get_husimi_factor_on_subspace(subspace_modes=subspace_modes)

        return hafnian(A_reduced) / (
            husimi_factor
            * np.prod(factorial(occupation_numbers))
        )

    samples = []

    for _ in repeat(None, shots):
        outcome = []

        previous_probability = 1.0

        for k in range(len(modes)):
            subspace_modes = tuple(modes[:(k + 1)])

            cumulated_probabilities = [0.0]

            guess = random.uniform(0.0, 1.0)

            choice = None

            for n in range(cutoff + 1):
                occupation_numbers = tuple(outcome + [n])

                probability = get_probability(
                    subspace_modes=subspace_modes,
                    occupation_numbers=occupation_numbers
                )
                conditional_probability = probability / previous_probability
                cumulated_probabilities.append(
                    conditional_probability + cumulated_probabilities[-1]
                )
                if guess < cumulated_probabilities[-1]:
                    choice = n
                    break

            else:
                choice = choose_from_cumulated_probabilities(cumulated_probabilities)

            previous_probability = (
                cumulated_probabilities[choice + 1]
                - cumulated_probabilities[choice]
            ) * previous_probability

            outcome.append(choice)

        samples.append(outcome)

    print("PQ OUTCOME", samples)

    return samples
