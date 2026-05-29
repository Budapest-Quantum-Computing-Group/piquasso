#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
import pytest

from piquasso.lxeb import lxe_ref_boson_sampling, lxe_ref_gaussian_boson_sampling


def test_lxe_ref_boson_sampling():
    assert np.isclose(lxe_ref_boson_sampling(n=10, m=3), 0.018798373343827918)


def test_lxe_ref_gaussian_boson_sampling():
    assert np.isclose(
        lxe_ref_gaussian_boson_sampling(n=10, m=12, d=5), 2.146772850363915e-05
    )


def test_lxe_ref_gaussian_boson_sampling_odd_n_is_zero():
    assert lxe_ref_gaussian_boson_sampling(n=11, m=12, d=5) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=10.0, m=3),
        dict(n=-1, m=3),
        dict(n=1, m=0),
    ],
)
def test_lxe_ref_boson_sampling_invalid_params(kwargs):
    with pytest.raises((TypeError, ValueError)):
        lxe_ref_boson_sampling(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=10.0, m=12, d=5),
        dict(n=-2, m=12, d=5),
        dict(n=2, m=0, d=1),
        dict(n=2, m=12, d=0),
        dict(n=2, m=12, d=13),
    ],
)
def test_lxe_ref_gaussian_boson_sampling_invalid_params(kwargs):
    with pytest.raises((TypeError, ValueError)):
        lxe_ref_gaussian_boson_sampling(**kwargs)
