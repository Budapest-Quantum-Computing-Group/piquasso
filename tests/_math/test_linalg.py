#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso._math.linalg import block_reduce


def test_block_reduce_on_2_by_2_matrix():
    matrix = np.array(
        [
            [11, 12],
            [21, 22],
        ],
        dtype=float,
    )

    result = block_reduce(matrix, reduce_on=(2,))

    assert np.allclose(
        result,
        np.array(
            [
                [11, 11, 12, 12],
                [11, 11, 12, 12],
                [21, 21, 22, 22],
                [21, 21, 22, 22],
            ]
        ),
    )


def test_block_reduce_on_4_by_4_matrix():
    matrix = np.array(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44],
        ],
        dtype=float,
    )

    result = block_reduce(matrix, reduce_on=(1, 2))

    assert np.allclose(
        result,
        np.array(
            [
                [11, 12, 12, 13, 14, 14],
                [21, 22, 22, 23, 24, 24],
                [21, 22, 22, 23, 24, 24],
                [31, 32, 32, 33, 34, 34],
                [41, 42, 42, 43, 44, 44],
                [41, 42, 42, 43, 44, 44],
            ],
        ),
    )


def test_hafnian_on_6_by_6_matrix():
    matrix = np.array(
        [
            [11, 12, 13, 14, 15, 16],
            [21, 22, 23, 24, 25, 26],
            [31, 32, 33, 34, 35, 36],
            [41, 42, 43, 44, 45, 46],
            [51, 52, 53, 54, 55, 56],
            [61, 62, 63, 64, 65, 66],
        ],
        dtype=float,
    )

    result = block_reduce(matrix, reduce_on=(1, 0, 2))

    assert np.allclose(
        result,
        np.array(
            [
                [11, 13, 13, 14, 16, 16],
                [31, 33, 33, 34, 36, 36],
                [31, 33, 33, 34, 36, 36],
                [41, 43, 43, 44, 46, 46],
                [61, 63, 63, 64, 66, 66],
                [61, 63, 63, 64, 66, 66],
            ],
        ),
    )
