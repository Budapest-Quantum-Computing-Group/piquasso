#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from scipy.special import factorial

from piquasso._math.hermite import hermite_kampe, hermite_kampe_2dim


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
