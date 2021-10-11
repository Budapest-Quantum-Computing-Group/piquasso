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

import abc

from typing import Optional, List, Type

from piquasso.api.result import Result
from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.program import Program
from piquasso.api.errors import InvalidParameter
from piquasso.api.instruction import Instruction


class Computer(abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        program: Program,
        shots: int = 1,
    ) -> Optional[Result]:
        pass


class Simulator(Computer, abc.ABC):
    state_class: Type[State]

    def __init__(self, d: int, config: Config = None) -> None:
        self.d = d
        self.config = config.copy() if config is not None else Config()

    @property
    @abc.abstractmethod
    def _instruction_map(self) -> dict:
        pass

    def create_initial_state(self):
        return self.state_class(d=self.d, config=self.config)

    def execute_instructions(
        self,
        instructions: List[Instruction],
        initial_state: State = None,
        shots: int = 1,
    ) -> State:
        if not isinstance(shots, int) or shots < 1:
            raise InvalidParameter(
                f"The number of shots should be a positive integer: shots={shots}."
            )

        state = initial_state.copy() if initial_state else self.create_initial_state()

        # TODO: This is not a nice solution.
        state.shots = shots

        for instruction in instructions:
            if not hasattr(instruction, "modes") or instruction.modes is tuple():
                instruction.modes = tuple(range(self.d))

            if hasattr(instruction, "_autoscale"):
                instruction._autoscale()  # type: ignore

            method_name = self._instruction_map.get(instruction.__class__.__name__)

            if not method_name:
                raise NotImplementedError(
                    "\n"
                    "No such instruction implemented for this state.\n"
                    "Details:\n"
                    f"instruction={instruction}\n"
                    f"state={state}\n"
                    f"Available instructions:\n"
                    + str(", ".join(self._instruction_map.keys()))
                    + "."
                )

            getattr(state, method_name)(instruction)

        return state

    def execute(
        self, program: Program, shots: int = 1, initial_state: State = None
    ) -> Optional[Result]:
        """Applies the given program to the state and executes it.

        Args:
            program (Program):
                The program whose instructions are used in the simpulation.
            shots (int):
                The number of samples to generate.
        """

        state = self.execute_instructions(
            program.instructions, initial_state=initial_state, shots=shots
        )

        return Result(
            samples=state.result.samples if state.result else [],
            state=state,
        )
