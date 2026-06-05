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

from piquasso._math.validations import (
    is_natural,
    all_natural,
    validate_occupation_numbers,
    validate_postselection_cutoff,
)
from piquasso.api.exceptions import InvalidState

import pytest


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


def test_validate_occupation_numbers_includes_cutoff_hint():
    with pytest.raises(InvalidState) as error:
        validate_occupation_numbers([0, 1, 0, 1, 1, 1], d=6, cutoff=4)

    assert "pq.Config(cutoff=5)" in error.value.args[0]


def test_validate_postselection_cutoff_when_postselection_consumes_cutoff():
    with pytest.raises(InvalidState) as error:
        validate_postselection_cutoff(
            cutoff=3,
            photon_counts=(1, 1, 1),
            occupation_numbers=[[1, 1, 1, 0]],
        )

    message = error.value.args[0]
    assert "Post-selecting 3 photon(s)" in message
    assert "pq.Config(cutoff=4)" in message


def test_validate_postselection_cutoff_when_remaining_photons_exceed_truncation():
    with pytest.raises(InvalidState) as error:
        validate_postselection_cutoff(
            cutoff=5,
            photon_counts=(0, 0),
            occupation_numbers=[[2, 1, 1, 1, 0, 0]],
        )

    message = error.value.args[0]
    assert "After post-selecting" in message
    assert "pq.Config(cutoff=6)" in message
