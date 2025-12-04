#
# Copyright 2021-2025 Budapest Quantum Computing Group
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
from typing import List, Type, Dict

import blackbird as bb

from ..api.instruction import Instruction
from ..api.exceptions import PiquassoException


_BB_TO_PQ_MAP = {
    "Dgate": "Displacement",
    "Xgate": "PositionDisplacement",
    "Zgate": "MomentumDisplacement",
    "Sgate": "Squeezing",
    "Pgate": "QuadraticPhase",
    "Kgate": "Kerr",
    "Rgate": "Phaseshifter",
    "BSgate": "Beamsplitter",
    "MZgate": "MachZehnder",
    "S2gate": "Squeezing2",
    "CXgate": "ControlledX",
    "CZgate": "ControlledZ",
    "CKgate": "CrossKerr",
    "Vgate": "CubicPhase",
    "Fouriergate": "Fourier",
}

_PQ_TO_BB_MAP = {v: k for k, v in _BB_TO_PQ_MAP.items()}


def load_instructions(blackbird_program: bb.BlackbirdProgram) -> List[Instruction]:
    """
    Loads the gates to apply into :attr:`Program.instructions` from a
    :class:`~blackbird.program.BlackbirdProgram`.

    Args:
        blackbird_program (~blackbird.program.BlackbirdProgram):
            The Blackbird program to use.
    """

    return [
        _blackbird_operation_to_instruction(operation)
        for operation in blackbird_program.operations
    ]


def export_instructions(instructions: List[Instruction]) -> bb.BlackbirdProgram:
    """
    Exports a list of :class:`~piquasso.api.instruction.Instruction` to a
    :class:`~blackbird.program.BlackbirdProgram`.

    Args:
        instructions (List[~piquasso.api.instruction.Instruction]):
            The list of instructions to export.

    Returns:
        ~blackbird.program.BlackbirdProgram:
            The exported Blackbird program.
    """

    blackbird_operations = [
        _piquasso_instruction_to_blackbird_operation(instruction)
        for instruction in instructions
    ]

    blackbird_program = bb.BlackbirdProgram(name="Exported Piquasso program")
    blackbird_program._operations = blackbird_operations
    if instructions:
        blackbird_program._modes = (
            max(mode for instruction in instructions for mode in instruction.modes) + 1
        )
    else:
        blackbird_program._modes = 0

    return blackbird_program


def _blackbird_operation_to_instruction(
    blackbird_operation: dict,
) -> Instruction:
    op = blackbird_operation["op"]
    pq_instruction_name = _BB_TO_PQ_MAP.get(op)

    pq_instruction_class = (
        Instruction.get_subclass(pq_instruction_name) if pq_instruction_name else None
    )

    if pq_instruction_class is None:
        raise PiquassoException(f"Operation {op} is not implemented in piquasso.")

    params = _get_instruction_params(
        pq_instruction_class=pq_instruction_class, bb_operation=blackbird_operation
    )

    instruction = pq_instruction_class(**params)

    instruction.modes = tuple(blackbird_operation["modes"])

    return instruction


def _get_instruction_params(
    pq_instruction_class: Type[Instruction], bb_operation: dict
) -> dict:
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


def _piquasso_instruction_to_blackbird_operation(
    instruction: Instruction,
) -> Dict[str, object]:
    operation: Dict[str, object] = {
        "op": _PQ_TO_BB_MAP.get(instruction.__class__.__name__),
        "args": list(instruction.params.values()),
        "kwargs": {},
        "modes": list(instruction.modes),
    }

    return operation
