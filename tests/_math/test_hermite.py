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

from scipy.special import factorial

from piquasso._math.hermite import (
    hermite_kampe,
    hermite_kampe_2dim,
    hermite_multidim,
)


def test_hermite_kampe():
    result = hermite_kampe(n=3, x=0.5, y=0.2)

    assert np.isclose(result, 0.725)


def test_hermite_kampe_first_argument():
    x = 1.2
    n = 3

    result = hermite_kampe(n=n, x=x, y=0)

    assert np.isclose(result, x ** n)


def test_hermite_kampe_secund_argument_even():
    y = 1.2
    n = 4

    result = hermite_kampe(n=n, x=0, y=y)

    assert np.isclose(
        result,
        y ** (n // 2) * factorial(n) / factorial(n // 2)
    )


def test_hermite_kampe_secund_argument_odd():
    y = 1.2
    n = 3

    result = hermite_kampe(n=n, x=0, y=y)

    assert np.isclose(result, 0.0)


def test_hermite_kampe_2dim():
    result = hermite_kampe_2dim(n=2, m=3, x=0.75, y=0.25, z=0.3, u=0.7, tau=1.5)

    assert np.isclose(result, 15.29859375)


def test_hermite_multidim_for_one_dimension_zeros():
    result = hermite_multidim(np.array([[1.0]]), [0], np.array([0.5]))

    assert np.isclose(result, 1.0)


def test_hermite_multidim_for_one_dimension_basis():
    result = hermite_multidim(np.array([[1.0]]), [1], np.array([0.5]))

    assert np.isclose(result, 0.5)


def test_hermite_multidim_for_two_dimension_zeros():
    result = hermite_multidim(
        B=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        n=[0, 0],
        alpha=np.array([0.5, 0.6]),
    )

    assert np.isclose(result, 1.0)


def test_hermite_multidim_for_two_dimension_basis():
    result = hermite_multidim(
        B=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        n=[0, 1],
        alpha=np.array([0.5, 0.6]),
    )

    assert np.isclose(result, 3.9)
