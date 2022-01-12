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

from piquasso.core import _mixins

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

from .computer import Computer


class Simulator(Computer, _mixins.CodeMixin):
    """Base class for all simulators defined in Piquasso."""

    _state_class: Type[State]
    _config_class: Type[Config] = Config

    def __init__(self, d: int, config: Optional[Config] = None) -> None:
        self.d = d
        self.config = config.copy() if config is not None else self._config_class()

    @property
    @abc.abstractmethod
    def _instruction_map(self) -> Dict[Type[Instruction], Callable]:
        pass

    def _as_code(self):
        if self.config == Config():
            return f"pq.{self.__class__.__name__}(d={self.d})"

        four_spaces = " " * 4
        return (
            f"pq.{self.__class__.__name__}(\n"
            f"{four_spaces}d={self.d}, config={self.config._as_code()}\n"
            ")"
        )

    def create_initial_state(self):
        """Creates an initial state with no instructions executed.

        Note: This not necessarily needs to be a vacuum state.

        Returns:
            State: The initial state of the simulation.
        """

        return self._state_class(d=self.d, config=self.config)

    def _validate_instruction_existence(self, instructions: List[Instruction]) -> None:
        for instruction in instructions:
            self._get_calculation(instruction)

    def _get_calculation(self, instruction: Instruction) -> Callable:
        for instruction_class, calculation in self._instruction_map.items():
            if type(instruction) == instruction_class:
                return calculation

        raise InvalidSimulation(
            "\n"
            "No such instruction implemented for this simulator.\n"
            "Details:\n"
            f"instruction={instruction}\n"
            f"simulator={self}\n"
            f"Available instructions:\n"
            + str(", ".join(map(repr, self._instruction_map.keys())))
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
        if not isinstance(initial_state, self._state_class):
            raise InvalidState(
                f"State specified with type '{type(initial_state)}', but it should be "
                f"{self._state_class} for this simulator."
            )

    def validate(self, program: Program) -> None:
        """Validates the specified program.

        Raises:
            InvalidInstruction: When invalid instructions are defined in the program.
            InvalidSimulation:
                When the instructions are valid, but the simulator couldn't execute the
                specified program.

        Args:
            program (Program): The program to validate.
        """

        self._validate_instructions(program.instructions)

    def execute_instructions(
        self,
        instructions: List[Instruction],
        initial_state: State = None,
        shots: int = 1,
    ) -> Result:
        """Executes the specified instruction list.

        Args:
            instructions (List[Instruction]): The instructions to execute.
            initial_state (State, optional):
                A state to execute the instructions on. Defaults to the state created by
                :meth:`create_initial_state`.
            shots (int, optional):
                The number of times the program should be execute. Defaults to 1.

        Raises:
            InvalidParameter: When `shots` is not a positive integer.

        Returns:
            Result:
                The result of the simulation containing the resulting state and samples
                if any measurement is specified in `instructions`.
        """

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

        result = Result(state=state)

        for instruction in instructions:
            if not hasattr(instruction, "modes") or instruction.modes is tuple():
                instruction.modes = tuple(range(self.d))

            if hasattr(instruction, "_autoscale"):
                instruction._autoscale()  # type: ignore

            calculation = self._get_calculation(instruction)

            result = calculation(result.state, instruction, shots)

        return result

    def execute(
        self, program: Program, shots: int = 1, initial_state: State = None
    ) -> Result:
        """Executes the specified program.

        Args:
            program (Program): The program to execute.
            initial_state (State, optional):
                A state to execute the instructions on. Defaults to the state created by
                :meth:`create_initial_state`.
            shots (int, optional):
                The number of times the program should execute. Defaults to 1.

        Raises:
            InvalidParameter: When `shots` is not a positive integer.

        Returns:
            Result:
                The result of the simulation containing the resulting state and samples
                if any measurement is specified in `program`.
        """

        instructions: List[Instruction] = program.instructions

        return self.execute_instructions(
            instructions, initial_state=initial_state, shots=shots
        )
