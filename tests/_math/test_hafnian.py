#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np

from piquasso._math.hafnian import hafnian


def test_hafnian_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 500.0)


def test_hafnian_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [3, 7, 3, 4],
            [4, 8, 4, 8],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 60.0)


def test_hafnian_on_6_by_6_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 6, 7, 8, 9, 5],
            [3, 7, 3, 4, 3, 7],
            [4, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2, 0],
            [6, 5, 7, 1, 0, 1],
        ],
        dtype=float,
    )

    assert np.isclose(hafnian(matrix), 1262.0)


@pytest.mark.monkey
def test_random_10_by_10_hafnian(generate_symmetric_matrix):
    matrix = generate_symmetric_matrix(10)

    assert hafnian(matrix)
