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

import piquasso as pq


def test_StateVector_is_valid_specifying_floats_close_to_integers():
    pq.StateVector([1, 1.0, 2.0, 0.0])


def test_StateVector_raises_InvalidState_when_nonintegers_specified():
    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        pq.StateVector([1, 1.3, 2.4])

    assert error.value.args[0] == (
        "Invalid occupation numbers: occupation_numbers=[1, 1.3, 2.4]\n"
        "Occupation numbers must contain non-negative integers."
    )
