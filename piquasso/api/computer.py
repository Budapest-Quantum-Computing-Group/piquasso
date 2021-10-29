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

from typing import Optional, List, Type, Dict, Callable

from piquasso.api.result import Result
from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.program import Program
from piquasso.api.errors import (
    InvalidParameter,
    InvalidInstruction,
    InvalidSimulation,
    InvalidState,
)
from piquasso.api.instruction import Gate, Instruction, Measurement, Preparation

from piquasso._math.lists import is_ordered_sublist, deduplicate_neighbours


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
    def _instruction_map(self) -> Dict[str, Callable]:
        pass

    def create_initial_state(self):
        return self.state_class(d=self.d, config=self.config)

    def _validate_instruction_existence(self, instructions: List[Instruction]) -> None:
        for instruction in instructions:
            if instruction.__class__.__name__ not in self._instruction_map:
                raise InvalidInstruction(
                    "\n"
                    "No such instruction implemented for this simulator.\n"
                    "Details:\n"
                    f"instruction={instruction}\n"
                    f"simulator={self}\n"
                    f"Available instructions:\n"
                    + str(", ".join(self._instruction_map.keys()))
                    + "."
                )

    def _validate_instruction_order(self, instructions: List[Instruction]) -> None:

        all_instruction_categories = [Preparation, Gate, Measurement]

        def _to_instruction_category(instruction: Instruction) -> Type[Instruction]:
            for instruction_category in all_instruction_categories:
                if isinstance(instruction, instruction_category):
                    return instruction_category

            raise InvalidInstruction(
                "\n"
                "The instruction is not a subclass of the following classes:\n"
                "{all_instruction_classes}.\n"
                "Make sure that all your instructions are subclassed properly from the "
                "above classes.\n"
                "instruction={instruction}."
            )

        instruction_category_projection = list(
            map(_to_instruction_category, instructions)
        )

        measurement_count = instruction_category_projection.count(Measurement)

        if measurement_count not in (0, 1):
            raise InvalidSimulation(
                "Up to one measurement could be registered for simulations."
            )

        if (
            measurement_count == 1
            and instruction_category_projection[-1] != Measurement
        ):
            raise InvalidSimulation(
                "Measurement should be registered only at the end of a program during "
                f"simulation: measurement={instruction_category_projection[-1]}."
            )

        current_instruction_categories_in_order = deduplicate_neighbours(
            instruction_category_projection,
        )

        if not is_ordered_sublist(
            current_instruction_categories_in_order,
            all_instruction_categories,
        ):
            raise InvalidSimulation(
                "The simulator could only execute instructions in the "
                "preparation-gate-measurement order."
            )

    def _validate_instructions(self, instructions: List[Instruction]) -> None:
        self._validate_instruction_existence(instructions)
        self._validate_instruction_order(instructions)

    def _validate_state(self, initial_state: State) -> None:
        if not isinstance(initial_state, self.state_class):
            raise InvalidState(
                f"State specified with type '{type(initial_state)}', but it should be "
                f"{self.state_class} for this simulator."
            )

    def validate(self, program):
        self._validate_instructions(program.instructions)

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

        self._validate_instructions(instructions)

        if initial_state is not None:
            self._validate_state(initial_state)
            state = initial_state.copy()
        else:
            state = self.create_initial_state()

        # TODO: This is not a nice solution.
        state.shots = shots

        for instruction in instructions:
            if not hasattr(instruction, "modes") or instruction.modes is tuple():
                instruction.modes = tuple(range(self.d))

            if hasattr(instruction, "_autoscale"):
                instruction._autoscale()  # type: ignore

            calculation = self._instruction_map[instruction.__class__.__name__]

            state = calculation(state, instruction)

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

        instructions: List[Instruction] = program.instructions

        state = self.execute_instructions(
            instructions, initial_state=initial_state, shots=shots
        )

        return Result(
            samples=state.result.samples if state.result else [],
            state=state,
        )
