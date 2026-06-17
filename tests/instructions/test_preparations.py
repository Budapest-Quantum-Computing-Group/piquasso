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

import pytest

import numpy as np

import piquasso as pq


def test_NumberState_is_valid_specifying_floats_close_to_integers():
    pq.NumberState([1, 1.0, 2.0, 0.0])


def test_NumberState_raises_InvalidState_when_nonintegers_specified():
    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        pq.NumberState([1, 1.3, 2.4])

    assert error.value.args[0] == (
        "Invalid occupation numbers: occupation_numbers=[1, 1.3, 2.4]\n"
        "Occupation numbers must contain non-negative integers."
    )


def test_FockStateVector_stores_fock_amplitude_map():
    amplitude_map = {(0,): 1.0, (1,): 0.0}

    preparation = pq.FockStateVector(fock_amplitude_map=amplitude_map)

    assert preparation.params["fock_amplitude_map"] == amplitude_map


def test_NumberState_raises_TypeError_when_no_arguments():
    with pytest.raises(TypeError) as error:
        pq.NumberState()

    assert error.value.args[0] == (
        "NumberState.__init__() missing 1 required positional argument: "
        "'occupation_numbers'"
    )


def test_StateVector_raises_InvalidParameter_when_fock_amplitude_map_given():
    amplitude_map = {(0,): 1.0}

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.StateVector([0], fock_amplitude_map=amplitude_map)

    assert error.value.args[0] == (
        "Only one of 'occupation_numbers' or 'fock_amplitude_map' can be provided."
    )


def test_FockStateVector_raises_InvalidState_when_fock_amplitude_map_keys_invalid():
    amplitude_map = {(0, 1.2): 1.0}

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        pq.FockStateVector(fock_amplitude_map=amplitude_map)

    assert error.value.args[0] == (
        "Invalid occupation numbers in fock_amplitude_map: {(0, 1.2): 1.0}\n"
        "Occupation numbers must contain non-negative integers."
    )


def test_NumberState_addition_preserves_coefficients():
    state_vector = (
        pq.NumberState([1], coefficient=2.0) + pq.NumberState([2], coefficient=3.0)
    ) * 5.0

    result = state_vector + pq.NumberState([3], coefficient=7.0)

    assert result.params["fock_amplitude_map"] == {(1,): 10.0, (2,): 15.0, (3,): 7.0}


def test_NumberState_addition_preserves_coefficients_with_fock_amplitude_map():
    state_vector = (
        pq.NumberState([1], coefficient=2.0) + pq.NumberState([2], coefficient=3.0)
    ) * 5.0

    result = state_vector + pq.FockStateVector(
        fock_amplitude_map={(3,): 7.0}, coefficient=1.0
    )

    assert result.params["fock_amplitude_map"] == {(1,): 10.0, (2,): 15.0, (3,): 7.0}


def test_FockStateVector_addition_preserves_coefficients_with_fock_amplitude_map():
    state_vector = (
        pq.FockStateVector(fock_amplitude_map={(1,): 2.0})
        + pq.FockStateVector(fock_amplitude_map={(2,): 3.0})
    ) * 5.0

    result = state_vector + pq.FockStateVector(
        fock_amplitude_map={(3,): 7.0}, coefficient=1.0
    )

    assert result.params["fock_amplitude_map"] == {(1,): 10.0, (2,): 15.0, (3,): 7.0}


def test_FockStateVector_addition_preserves_coefficients_with_NumberState():
    state_vector = (
        pq.FockStateVector(fock_amplitude_map={(1,): 2.0})
        + pq.FockStateVector(fock_amplitude_map={(2,): 3.0})
    ) * 5.0

    result = state_vector + pq.NumberState([3], coefficient=7.0)

    assert result.params["fock_amplitude_map"] == {(1,): 10.0, (2,): 15.0, (3,): 7.0}


def test_NumberState_adding_same_occupation_numbers_preserves_coefficients():
    state_vector = (
        pq.NumberState([1], coefficient=2.0) + pq.NumberState([1], coefficient=3.0)
    ) * 5.0

    result = state_vector + pq.NumberState([1], coefficient=7.0)

    assert result.params["occupation_numbers"] == (1,)
    assert np.isclose(result.params["coefficient"], 2.0 * 5.0 + 3.0 * 5.0 + 7.0)


def test_FockStateVector_adding_same_occupation_numbers_preserves_coefficients():
    state_vector = (
        pq.FockStateVector(fock_amplitude_map={(1,): 2.0})
        + pq.FockStateVector(fock_amplitude_map={(1,): 3.0})
    ) * 5.0

    result = state_vector + pq.FockStateVector(
        fock_amplitude_map={(1,): 7.0, (2,): 8.0}, coefficient=1.0
    )

    assert result.params["fock_amplitude_map"] == {(1,): 5.0 * 5.0 + 7.0, (2,): 8.0}


def test_FockStateVector_adding_same_occ_numbers_preserves_coeffs_with_NumberState():
    state_vector = (
        pq.FockStateVector(fock_amplitude_map={(1,): 2.0})
        + pq.FockStateVector(fock_amplitude_map={(1,): 3.0})
    ) * 5.0

    result = state_vector + pq.NumberState([1], coefficient=7.0)

    assert result.params["fock_amplitude_map"] == {(1,): 5.0 * 5.0 + 7.0}


def test_NumberState_adding_same_occ_numbers_preserves_coeffs_with_FockStateVector():
    state_vector = (
        pq.NumberState([1], coefficient=2.0) + pq.NumberState([1], coefficient=3.0)
    ) * 5.0

    result = state_vector + pq.FockStateVector(
        fock_amplitude_map={(1,): 7.0, (2,): 8.0}, coefficient=1.0
    )

    assert result.params["fock_amplitude_map"] == {(1,): 5.0 * 5.0 + 7.0, (2,): 8.0}
