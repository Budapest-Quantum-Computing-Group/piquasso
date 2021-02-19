#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

from piquasso import functions


@pytest.fixture
def d():
    return 1


@pytest.fixture
def mean():
    return np.array([1, 2])


@pytest.fixture
def cov():
    return np.array(
        [
            [1, 2],
            [-3, 4],
        ]
    )


def test_wigner_function_at_scalar(d, mean, cov):
    quadrature_array = np.array([1, 2])

    expected = 0.10065842420897406

    actual = functions._gaussian_wigner_function_for_scalar(
        X=quadrature_array, d=d, mean=mean, cov=cov
    )

    assert np.allclose(expected, actual)


def test_gaussian_wigner_function_handles_vectors(d, mean, cov):
    quadrature_matrix = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    expected = np.array(
        [
            0.10065842420897406,
            0.009131526225575573,
            6.81746788883418e-06,
        ],
    )

    actual = functions.gaussian_wigner_function(
        quadrature_matrix, d=d, mean=mean, cov=cov
    )

    assert np.allclose(expected, actual)
