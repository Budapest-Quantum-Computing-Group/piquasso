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

from typing import Optional, List, Type

from piquasso.api.config import Config
from piquasso.api.calculator import BaseCalculator

from piquasso.api.simulator import Simulator
from piquasso.api.exceptions import InvalidSimulation

from .calculator import _BuiltinCalculator


class BuiltinSimulator(Simulator):
    _extra_builtin_calculators: List[Type[BaseCalculator]] = []

    def __init__(
        self,
        d: int,
        config: Optional[Config] = None,
        calculator: Optional[BaseCalculator] = None,
    ) -> None:
        if calculator is not None:
            self._validate_calculator(calculator)

        super().__init__(d=d, config=config, calculator=calculator)

    def _validate_calculator(self, calculator):
        if (
            isinstance(calculator, _BuiltinCalculator)
            and not isinstance(calculator, self._default_calculator_class)
            and not any(
                [isinstance(calculator, cls) for cls in self._extra_builtin_calculators]
            )
        ):
            raise InvalidSimulation(
                f"The calculator '{calculator}' is not supported."
                f"Supported calculators:"
                "\n"
                f"{[self._default_calculator_class] + self._extra_builtin_calculators}"
            )
