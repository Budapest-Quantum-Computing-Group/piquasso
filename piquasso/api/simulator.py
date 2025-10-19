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
import numpy as np

from typing import Optional, List, Type, Dict, Callable, Tuple

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
    _measurement_classes_allowed_mid_circuit: Tuple[Type[Measurement], ...] = tuple()

    def __init__(
        self,
        d: Optional[int] = None,
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
        if self.d is None:
            d_param = ""
        else:
            d_param = f"d={self.d}"

        if self.config == Config():
            if d_param:
                return f"pq.{self.__class__.__name__}({d_param})"
            else:
                return f"pq.{self.__class__.__name__}()"

        four_spaces = " " * 4
        if d_param:
            return (
                f"pq.{self.__class__.__name__}(\n"
                f"{four_spaces}{d_param}, config={self.config._as_code()}\n"
                ")"
            )
        else:
            return (
                f"pq.{self.__class__.__name__}(\n"
                f"{four_spaces}config={self.config._as_code()}\n"
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
        # At this point, d must be set (either explicitly or inferred)
        assert self.d is not None, "d must be set before validation"

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

    def _validate_measurements_at_end(self, instructions):
        for index, instruction in enumerate(instructions):
            is_measurement = isinstance(instruction, Measurement)

            if (
                is_measurement
                and index != len(instructions) - 1
                and not isinstance(
                    instruction, self._measurement_classes_allowed_mid_circuit
                )
            ):
                raise InvalidSimulation(
                    f"Measurement {instruction} is not allowed as a mid-circuit "
                    f"measurement for this simulator."
                    f"Allowed mid-circuit measurement classes:"
                    f"{self._measurement_classes_allowed_mid_circuit}."
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

    def _is_instruction_list_multi_measurement(self, instructions):
        count = 0

        for instruction in instructions:
            if isinstance(instruction, Measurement):
                count += 1
                if count > 1:
                    return True

        return False

    @staticmethod
    def _remap_modes(active_modes, modes_to_remap):
        return tuple(active_modes.index(mode) for mode in modes_to_remap)

    @staticmethod
    def _remap_modes_inverse(active_modes, modes_to_remap):
        return tuple(active_modes[mode] for mode in modes_to_remap)

    @staticmethod
    def _delete_modes_from_active(active_modes, modes):
        return tuple(
            mode
            for mode in active_modes
            if mode not in Simulator._remap_modes_inverse(active_modes, modes)
        )

    def _execute_multi_measurement_instructions(self, state, instructions, shots):
        samples = []

        for _ in range(shots):
            active_modes = tuple(range(self.d))

            result = Result(state=state.copy())

            samples_for_this_shot = []

            for instruction_orig in instructions:
                instruction = instruction_orig.copy()

                if not hasattr(instruction, "modes") or instruction.modes is tuple():
                    instruction.modes = active_modes

                if any(m not in active_modes for m in instruction.modes):
                    inactive_modes = {
                        m for m in instruction.modes if m not in active_modes
                    }
                    raise ValueError(
                        f"Some modes of instruction {instruction} are not active: "
                        f"{inactive_modes}."
                    )

                instruction.modes = Simulator._remap_modes(
                    active_modes, instruction.modes
                )

                calculation = self._get_calculation(instruction)

                instruction = self._maybe_postprocess_batch_instruction(instruction)

                result = calculation(result.state, instruction, shots=1)

                if (
                    isinstance(result.samples, np.ndarray)
                    and result.samples.size > 0
                    or result.samples
                ):
                    samples_for_this_shot.extend(result.samples[0])

                if isinstance(instruction, Measurement):
                    active_modes = Simulator._delete_modes_from_active(
                        active_modes, instruction.modes
                    )

            samples.append(samples_for_this_shot)

        return Result(samples=samples, state=result.state)

    def execute_instructions(
        self,
        instructions: List[Instruction],
        initial_state: Optional[State] = None,
        shots: int = 1,
        program: Optional[Program] = None,
    ) -> Result:
        """Executes the specified instruction list.

        Args:
            instructions (List[Instruction]): The instructions to execute.
            initial_state (State, optional):
                A state to execute the instructions on. Defaults to the state created by
                :meth:`create_initial_state`.
            shots (int, optional):
                The number of times the program should be execute. Defaults to 1.
            program (Program, optional):
                The program containing the instructions. Used for mode inference when
                `d` is not set. Defaults to None.

        Raises:
            InvalidParameter: When `shots` is not a positive integer.
            InvalidSimulation: When the number of modes cannot be inferred and `d` was
                not provided during simulator initialization.

        Returns:
            Result:
                The result of the simulation containing the resulting state and samples
                if any measurement is specified in `instructions`.
        """

        if not isinstance(shots, int) or shots < 1:
            raise InvalidParameter(
                f"The number of shots should be a positive integer: shots={shots}."
            )

        # Infer d from program if not explicitly set
        if self.d is None:
            if program is None:
                raise InvalidSimulation(
                    "Cannot infer the number of modes: program not provided. "
                    "Please provide the 'd' parameter when creating the "
                    "simulator, or use execute() method with a Program instead "
                    "of execute_instructions()."
                )
            inferred_d = program.get_number_of_modes()
            if inferred_d is None:
                raise InvalidSimulation(
                    "Cannot infer the number of modes from the program. "
                    "Please provide the 'd' parameter when creating the simulator, "
                    "or ensure that at least one instruction in the program "
                    "specifies modes explicitly."
                )
            self.d = inferred_d

        self._validate_instructions(instructions)

        if initial_state is not None:
            self._validate_initial_state(initial_state)
            state = initial_state.copy()
        else:
            state = self.create_initial_state()

        is_multi_measurement = self._is_instruction_list_multi_measurement(instructions)

        if is_multi_measurement:
            return self._execute_multi_measurement_instructions(
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
            InvalidSimulation: When the number of modes cannot be inferred from the
                program and `d` was not provided during simulator initialization.

        Returns:
            Result:
                The result of the simulation containing the resulting state and samples
                if any measurement is specified in `program`.
        """

        instructions: List[Instruction] = program.instructions

        return self.execute_instructions(
            instructions, initial_state=initial_state, shots=shots, program=program
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, config={self.config}, connector={self._connector})"  # noqa: E501
