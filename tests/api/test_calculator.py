#
# Copyright 2021-2022 Budapest Quantum Computing Group
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
def dummy_diagonal():
    return np.array([0.0])


@pytest.fixture
def dummy_occupation_number():
    return (0,)


def test_BaseCalculator_raises_NotImplementedCalculation_for_permanent(
    dummy_matrix, dummy_occupation_number
):
    class PluginCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

    calculator = PluginCalculator()

    with pytest.raises(NotImplementedCalculation):
        calculator.permanent(
            dummy_matrix, dummy_occupation_number, dummy_occupation_number
        )


def test_BaseCalculator_raises_NotImplementedCalculation_for_hafnian(
    dummy_matrix, dummy_occupation_number
):
    class PluginCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

    calculator = PluginCalculator()

    with pytest.raises(NotImplementedCalculation):
        calculator.hafnian(dummy_matrix, dummy_occupation_number)


def test_BaseCalculator_raises_NotImplementedCalculation_for_loop_hafnian(
    dummy_matrix, dummy_diagonal, dummy_occupation_number
):
    class PluginCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

    calculator = PluginCalculator()

    with pytest.raises(NotImplementedCalculation):
        calculator.loop_hafnian(dummy_matrix, dummy_diagonal, dummy_occupation_number)


def test_BaseCalculator_with_overriding_defaults():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `BaseCalculator` class defaults need to be able to overridden for
    any plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_permanent():
        return 42

    def plugin_loop_hafnian():
        return 43

    class PluginCalculator(pq.api.calculator.BaseCalculator):
        def __init__(self) -> None:
            super().__init__()

            self.permanent = plugin_permanent
            self.loop_hafnian = plugin_loop_hafnian

    plugin_calculator = PluginCalculator()

    assert plugin_calculator.permanent is plugin_permanent
    assert plugin_calculator.loop_hafnian is plugin_loop_hafnian

    assert plugin_calculator.permanent() == 42
    assert plugin_calculator.loop_hafnian() == 43
