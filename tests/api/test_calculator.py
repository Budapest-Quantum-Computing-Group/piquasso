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

import numpy as np

import pytest

import piquasso as pq
from piquasso.api.exceptions import NotImplementedCalculation


@pytest.fixture
def dummy_matrix():
    return np.array([[0.0]])


@pytest.fixture
def dummy_array():
    return np.array([0.0])


@pytest.fixture
def dummy_occupation_number():
    return (0,)


@pytest.fixture
def empty_calculator():
    class EmptyCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

    return EmptyCalculator()


def test_BaseCalculator_raises_NotImplementedCalculation_for_hafnian(
    empty_calculator,
    dummy_matrix,
    dummy_occupation_number,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.hafnian(dummy_matrix, dummy_occupation_number)


def test_BaseCalculator_raises_NotImplementedCalculation_for_loop_hafnian(
    empty_calculator, dummy_matrix, dummy_array, dummy_occupation_number
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.loop_hafnian(
            dummy_matrix, dummy_array, dummy_occupation_number
        )


def test_BaseCalculator_raises_NotImplementedCalculation_for_assign(
    empty_calculator,
    dummy_array,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.assign(dummy_array, index=0, value=3)


def test_BaseCalculator_raises_NotImplementedCalculation_for_scatter(
    empty_calculator,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.scatter(indices=[], updates=[], shape=(3, 3))


def test_BaseCalculator_raises_NotImplementedCalculation_for_block(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.block(
            [[dummy_matrix, dummy_matrix], [dummy_matrix, dummy_matrix]]
        )


def test_BaseCalculator_raises_NotImplementedCalculation_for_block_diag(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.block_diag(dummy_matrix, dummy_matrix)


def test_BaseCalculator_raises_NotImplementedCalculation_for_polar(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.polar(dummy_matrix, side="left")


def test_BaseCalculator_raises_NotImplementedCalculation_for_logm(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.logm(dummy_matrix)


def test_BaseCalculator_raises_NotImplementedCalculation_for_expm(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.expm(dummy_matrix)


def test_BaseCalculator_raises_NotImplementedCalculation_for_powm(
    empty_calculator,
    dummy_matrix,
):
    with pytest.raises(NotImplementedCalculation):
        empty_calculator.powm(dummy_matrix, 42)


def test_BaseCalculator_with_overriding_defaults():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `BaseCalculator` class defaults need to be able to overridden for
    any plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_loop_hafnian():
        return 43

    class PluginCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

            self.loop_hafnian = plugin_loop_hafnian

    plugin_calculator = PluginCalculator()

    assert plugin_calculator.loop_hafnian is plugin_loop_hafnian

    assert plugin_calculator.loop_hafnian() == 43
