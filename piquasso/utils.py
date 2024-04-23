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

from typing import List

from ._math.indices import get_operator_index

from .instructions.gates import _PassiveLinearGate, Interferometer

from .api.calculator import BaseCalculator
from .api.config import Config
from .api.exceptions import PiquassoException


def multiply_passive_linear_gates(
    gates: List[_PassiveLinearGate],
    calculator: BaseCalculator,
    config: Config,
) -> Interferometer:
    modes_as_set = set()

    for gate in gates:
        if not isinstance(gate, _PassiveLinearGate):
            raise PiquassoException(f"Invalid gate specified: {gate}")

        modes_as_set.update(set(gate.modes))

    d = len(modes_as_set)

    modes = list(modes_as_set)
    modes.sort()

    interferometer_matrix = calculator.np.identity(d, dtype=config.dtype)

    for gate in gates:
        passive_block = gate._get_passive_block(calculator, config)

        index = []
        for mode in gate.modes:
            index.append(modes.index(mode))

        indices = get_operator_index(index)

        embedded_passive_block = calculator.embed_in_identity(passive_block, indices, d)

        interferometer_matrix = embedded_passive_block @ interferometer_matrix

    return Interferometer(interferometer_matrix).on_modes(*modes)
