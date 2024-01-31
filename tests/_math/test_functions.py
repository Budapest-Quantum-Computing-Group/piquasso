#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

from piquasso._math.functions import (
    gaussian_wigner_function,
    gaussian_wigner_function_for_scalar,
)


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

    actual = gaussian_wigner_function_for_scalar(
        X=quadrature_array, d=d, mean=mean, cov=cov
    )

    assert np.allclose(expected, actual)


def test_gaussian_wigner_function_handles_vectors(d, mean, cov):
    positions = [[1.0], [3.0], [5.0]]
    momentums = [[2.0], [4.0], [6.0]]

    expected = np.array(
        [
            [0.10065842420897406, 0.020322585354620785, 0.00016724973685064803],
            [0.0674733595496344, 0.009131526225575573, 5.0374652683254064e-05],
            [0.020322585354620785, 0.001843623348920587, 6.81746788883418e-06],
        ]
    )

    actual = gaussian_wigner_function(positions, momentums, d=d, mean=mean, cov=cov)

    assert np.allclose(expected, actual)
