#
# Copyright 2021 Budapest Quantum Computing Group
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

import numpy as np

from piquasso._math.hermite import modified_hermite_multidim


def test_modified_hermite_multidim_for_one_dimension_zeros():
    result = modified_hermite_multidim(np.array([[1.0]]), [0], np.array([0.5]))

    assert np.isclose(result, 1.0)


def test_modified_hermite_multidim_for_one_dimension_basis():
    B = np.array([[1.0]])

    n = [1]

    alpha = np.array([0.5])

    result = modified_hermite_multidim(B, n, alpha)

    assert np.isclose(result, 0.5)


def test_modified_hermite_multidim_for_two_dimension_zeros():
    B = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )
    n = [0, 0]
    alpha = np.array([0.5, 0.6])

    result = modified_hermite_multidim(
        B=B,
        n=n,
        alpha=alpha,
    )

    assert np.isclose(result, 1.0)


def test_modified_hermite_multidim_for_two_dimension_basis():
    B = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )

    n = [0, 2]

    alpha = np.array([0.5, 0.6])

    result = modified_hermite_multidim(
        B=B,
        n=n,
        alpha=alpha,
    )

    assert np.isclose(result, -3.64)
