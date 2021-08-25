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

"""Implementation of circuits."""

import abc
import typing
from typing import List, Optional

from .instruction import Instruction
from .result import Result

if typing.TYPE_CHECKING:
    from piquasso.api.state import State


class Circuit(abc.ABC):
    instruction_map: dict

    def __init__(self) -> None:
        self.result: Result = None  # type: ignore
        self.shots: int = None  # type: ignore

    def execute_instructions(
        self, instructions: List[Instruction], state: "State", shots: int
    ) -> Optional[Result]:
        """Executes the collected instructions in order.

        Raises:
            NotImplementedError:
                If no such method is implemented on the `Circuit` class.

        Args:
            instructions (list):
                The methods along with keyword arguments of the current circuit to be
                executed in order.
        """
        self.shots = shots

        for instruction in instructions:
            if instruction.modes is tuple():
                instruction.modes = tuple(range(state.d))

            if hasattr(instruction, "_autoscale"):
                instruction._autoscale()  # type: ignore

            method_name = self.instruction_map.get(instruction.__class__.__name__)

            if not method_name:
                raise NotImplementedError(
                    "\n"
                    "No such instruction implemented for this state.\n"
                    "Details:\n"
                    f"instruction={instruction}\n"
                    f"state={state}\n"
                    f"Available instructions:\n"
                    + str(", ".join(self.instruction_map.keys())) + "."
                )

            getattr(self, method_name)(instruction, state)

        return self.result
