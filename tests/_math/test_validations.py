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

from piquasso._math.validations import is_natural, all_natural


def test_zero_is_natural():
    assert is_natural(0)


def test_positive_integers_are_natural():
    assert is_natural(2)


def test_negative_integers_are_natural():
    assert not is_natural(-2)


def test_floats_close_to_integers_count_as_natural():
    assert is_natural(2.0)


def test_floats_NOT_close_to_integers_do_NOT_count_as_natural():
    assert not is_natural(2.5)


def test_all_natural_positive_case():
    assert all_natural([1, 1.0, 0.0, 2.0])


def test_all_natural_negative_case():
    assert not all_natural([1, 1.0, 0.0, -2.0])
