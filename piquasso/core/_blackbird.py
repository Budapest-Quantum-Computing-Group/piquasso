#
# Copyright 2021 Budapest Quantum Computing Group
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

import inspect
from collections import OrderedDict

from . import registry


def load_instructions(blackbird_program):
    """
    Loads the gates to apply into :attr:`Program.instructions` from a
    :class:`blackbird.BlackbirdProgram`

    Args:
        blackbird_program (blackbird.BlackbirdProgram): The BlackbirdProgram to use.
    """

    instruction_map = {
        "Dgate": registry._retrieve_class("Displacement"),
        "Xgate": registry._retrieve_class("PositionDisplacement"),
        "Zgate": registry._retrieve_class("MomentumDisplacement"),
        "Sgate": registry._retrieve_class("Squeezing"),
        "Pgate": registry._retrieve_class("QuadraticPhase"),
        "Vgate": None,
        "Kgate": registry._retrieve_class("Kerr"),
        "Rgate": registry._retrieve_class("Phaseshifter"),
        "BSgate": registry._retrieve_class("Beamsplitter"),
        "MZgate": registry._retrieve_class("MachZehnder"),
        "S2gate": registry._retrieve_class("Squeezing2"),
        "CXgate": registry._retrieve_class("ControlledX"),
        "CZgate": registry._retrieve_class("ControlledZ"),
        "CKgate": registry._retrieve_class("CrossKerr"),
        "Fouriergate": registry._retrieve_class("Fourier"),
    }

    return [
        _blackbird_operation_to_instruction(instruction_map, operation)
        for operation in blackbird_program.operations
    ]


def _blackbird_operation_to_instruction(instruction_map, blackbird_operation):
    pq_instruction_class = instruction_map.get(blackbird_operation["op"])

    params = _get_instruction_params(
        pq_instruction_class=pq_instruction_class, bb_operation=blackbird_operation
    )

    instruction = pq_instruction_class(**params)

    instruction.modes = tuple(blackbird_operation["modes"])

    return instruction


def _get_instruction_params(pq_instruction_class, bb_operation):
    bb_params = bb_operation.get("args", None)

    if bb_params is None:
        return {}

    parameters = inspect.signature(pq_instruction_class).parameters

    instruction_params = OrderedDict()

    for param_name, param in parameters.items():
        if param_name == "self":
            continue

        instruction_params[param_name] = param.default

    for pq_param_name, bb_param in zip(instruction_params.keys(), bb_params):
        instruction_params[pq_param_name] = bb_param

    return instruction_params