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

import piquasso as pq


def test_Calculator_with_overriding_defaults():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `Calculator` class defaults need to be able to overridden for any
    plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_permanent_function():
        return 42

    def plugin_loop_hafnian_function():
        return 43

    class PluginCalculator(pq.api.calculator.Calculator):
        def __init__(
            self,
            permanent_function=plugin_permanent_function,
            loop_hafnian_function=plugin_loop_hafnian_function,
            **kwargs
        ) -> None:
            super().__init__(
                permanent_function=permanent_function,
                loop_hafnian_function=loop_hafnian_function,
                **kwargs,
            )

    plugin_calculator = PluginCalculator()

    assert plugin_calculator.permanent_function is plugin_permanent_function
    assert plugin_calculator.loop_hafnian_function is plugin_loop_hafnian_function

    assert plugin_calculator.permanent_function() == 42
    assert plugin_calculator.loop_hafnian_function() == 43


def test_Calculator_subclass_defaults_can_be_overridden_by_user():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `Calculator` class defaults need to be able to overridden for any
    plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_permanent_function():
        return 42

    def user_defined_permanent_function():
        return 44

    class PluginCalculator(pq.api.calculator.Calculator):
        def __init__(
            self, permanent_function=plugin_permanent_function, **kwargs
        ) -> None:
            super().__init__(
                permanent_function=permanent_function,
                **kwargs,
            )

    plugin_calculator = PluginCalculator(
        permanent_function=user_defined_permanent_function
    )

    assert plugin_calculator.permanent_function is not plugin_permanent_function

    assert plugin_calculator.permanent_function is user_defined_permanent_function
