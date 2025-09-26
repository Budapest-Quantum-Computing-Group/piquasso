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

import abc

from typing import Optional, List, Type, Dict, Callable, Set

from piquasso.core import _mixins

from piquasso.api.result import Result
from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.program import Program
from piquasso.api.connector import BaseConnector
from piquasso.api.exceptions import (
    InvalidParameter,
    InvalidSimulation,
    InvalidState,
    InvalidModes,
)
from piquasso.api.instruction import (
    Instruction,
    Measurement,
    Preparation,
    BatchInstruction,
)

from .computer import Computer


class Simulator(Computer, _mixins.CodeMixin):
    """Base class for all simulators defined in Piquasso."""

    _state_class: Type[State]
    _config_class: Type[Config] = Config
    _default_connector_class: Type[BaseConnector]
    _admissible_measurement_classes: Set[Measurement] = set()

    def __init__(
        self,
        d: int,
        config: Optional[Config] = None,
        connector: Optional[BaseConnector] = None,
    ) -> None:
        self.d = d
        self.config = config.copy() if config is not None else self._config_class()
        self._connector = connector or self._default_connector_class()

    @property
    @abc.abstractmethod
    def _instruction_map(self) -> Dict[Type[Instruction], Callable]:
        """The map which associates an `Instruction` to a calculation function."""

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

        Note:
            This is not necessarily a vacuum state.

        Returns:
            State: The initial state of the simulation.
        """

        return self._state_class(
            d=self.d, connector=self._connector, config=self.config
        )

    def _validate_instruction_existence(self, instructions: List[Instruction]) -> None:
        for instruction in instructions:
            self._get_calculation(instruction)

    def _validate_instruction_modes(self, instructions: List[Instruction]) -> None:
        for instruction in instructions:
            if not instruction.modes:
                continue

            for mode in instruction.modes:
                if mode < 0 or mode >= self.d:
                    if self.d > 1:
                        valid_indices_message = (
                            f"Valid mode indices are between '0' and "
                            f"'{self.d - 1}' (inclusive)."
                        )
                    else:
                        valid_indices_message = (
                            "For a single-mode system, "
                            "the only valid mode index is '0'."
                        )
                    raise InvalidModes(
                        f"Instruction '{instruction}' addresses mode '{mode}',"
                        f" which is out of range "
                        f"for the simulator defined on '{self.d}' modes. "
                        f"{valid_indices_message}"
                    )

    def _get_calculation(self, instruction: Instruction) -> Callable:
        for instruction_class, calculation in self._instruction_map.items():
            if type(instruction) is instruction_class:
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

    def _validate_preparations_at_beginning(self, instructions):
        for index, instruction in enumerate(instructions):
            if isinstance(instruction, Preparation):
                previous_instuctions = instructions[:index]

                if any(
                    not isinstance(previous_instruction, Preparation)
                    for previous_instruction in previous_instuctions
                ):
                    raise InvalidSimulation(
                        f"Preparations should only be registered at the beginning of a "
                        f"program: instruction={instruction}."
                    )

    def _is_admissible_measurement(self, measurement):
        return any(
            isinstance(measurement, admissible_measurement_class)
            for admissible_measurement_class in self._admissible_measurement_classes
        )

    def _validate_measurements_at_end(self, instructions):
        for index, instruction in enumerate(instructions):
            is_measurement = isinstance(instruction, Measurement)

            if is_measurement and not self._is_admissible_measurement(instruction):
                next_instuctions = instructions[(index + 1) :]

                if any(
                    not isinstance(next_instruction, Measurement)
                    for next_instruction in next_instuctions
                ):
                    raise InvalidSimulation(
                        f"Measurements should only be registered at the end of a "
                        f"program: instruction={instruction}."
                        f"Admissible measurement classes:"
                        f"{self._admissible_measurement_classes}."
                    )

    def _validate_instruction_order(self, instructions: List[Instruction]) -> None:
        self._validate_preparations_at_beginning(instructions)

        self._validate_measurements_at_end(instructions)

    def _validate_instructions(self, instructions: List[Instruction]) -> None:
        self._validate_instruction_existence(instructions)
        self._validate_instruction_modes(instructions)
        self._validate_instruction_order(instructions)

    def _validate_initial_state(self, initial_state: State) -> None:
        if not isinstance(initial_state, self._state_class):
            raise InvalidState(
                f"Initial state is specified with type '{type(initial_state)}', but it "
                f"should be {self._state_class} for this simulator."
            )

        if initial_state.d != self.d:
            raise InvalidState(
                f"Mismatch in number of specified modes: According to the simulator, "
                f"the number of modes should be '{self.d}', but the specified "
                f"'initial_state' is defined on '{initial_state.d}' modes."
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

    def _maybe_postprocess_batch_instruction(
        self, instruction: Instruction
    ) -> Instruction:
        if isinstance(instruction, BatchInstruction):
            instruction._extra_params["execute"] = self.execute

        return instruction

    def _detect_many_measurement_scenario(self, instructions):
        count = 0

        for instruction in instructions:
            if isinstance(instruction, Measurement):
                count += 1
                if count > 1:
                    return True

        return False

    def _postprocess_many_measurement_sample(self, samples_on_modes):
        unrolled_samples_on_modes = {}
        for modes, subsample in samples_on_modes.items():
            for mode, value in zip(modes, subsample):
                unrolled_samples_on_modes[mode] = value

        return [value for _, value in sorted(unrolled_samples_on_modes.items())]

    def _execute_many_measurement_instructions(self, state, instructions, shots):
        samples = []

        def remap_modes(modes):
            return tuple(active_modes.index(mode) for mode in modes)

        def remap_modes_inverse(modes):
            return tuple(active_modes[mode] for mode in modes)

        def delete_modes_from_active(modes):
            return tuple(
                [
                    mode
                    for mode in active_modes
                    if mode not in remap_modes_inverse(modes)
                ]
            )

        for _ in range(shots):
            active_modes = tuple(range(self.d))

            result = Result(state=state.copy())

            samples_on_modes = {}

            for instruction_orig in instructions:
                instruction = instruction_orig.copy()

                if not hasattr(instruction, "modes") or instruction.modes is tuple():
                    instruction.modes = active_modes

                instruction.modes = remap_modes(instruction.modes)

                calculation = self._get_calculation(instruction)

                instruction = self._maybe_postprocess_batch_instruction(instruction)

                result = calculation(result.state, instruction, shots=1)

                if result.samples:
                    samples_on_modes[remap_modes_inverse(instruction.modes)] = (
                        result.samples[0]
                    )

                if isinstance(instruction, Measurement):
                    active_modes = delete_modes_from_active(instruction.modes)

            sample = self._postprocess_many_measurement_sample(samples_on_modes)

            samples.append(sample)

        return Result(samples=samples, state=result.state)

    def execute_instructions(
        self,
        instructions: List[Instruction],
        initial_state: Optional[State] = None,
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
            self._validate_initial_state(initial_state)
            state = initial_state.copy()
        else:
            state = self.create_initial_state()

        is_many_measurement = self._detect_many_measurement_scenario(instructions)

        if is_many_measurement:
            return self._execute_many_measurement_instructions(
                state, instructions, shots
            )

        result = Result(state=state)

        for instruction in instructions:
            if not hasattr(instruction, "modes") or instruction.modes is tuple():
                instruction.modes = tuple(range(self.d))

            calculation = self._get_calculation(instruction)

            instruction = self._maybe_postprocess_batch_instruction(instruction)

            result = calculation(result.state, instruction, shots)

        return result

    def execute(
        self, program: Program, shots: int = 1, initial_state: Optional[State] = None
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, config={self.config}, connector={self._connector})"  # noqa: E501
