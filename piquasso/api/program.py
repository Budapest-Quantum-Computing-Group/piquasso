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

from typing import List, Tuple, Any, Optional

import blackbird
from piquasso.core import _context, _blackbird
from piquasso.core import _mixins
from .instruction import Instruction
from .mode import Q


class Program(_mixins.DictMixin, _mixins.RegisterMixin, _mixins.CodeMixin):
    r"""The class representing the quantum program.

    A `Program` object can be used with the `with` statement.
    In this context all the instructions could be specified.

    Example usage::

        import numpy as np
        import piquasso as pq

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0, 1) | pq.Squeezing(r=0.5)

        simulator = pq.GaussianSimulator(d=5)
        result = simulator.execute(program)
    """

    instructions: list

    def __init__(
        self,
        instructions: Optional[list] = None,
    ) -> None:
        """
        Args:
            instructions (list[~piquasso.api.instruction.Instruction], optional):
                The set of instructions, e.g. quantum gates and measurements.
        """

        self.instructions: List[Instruction] = instructions or []

    @staticmethod
    def _map_modes(register: Q, instruction: Instruction) -> Tuple[int, ...]:
        if len(register.modes) == 0:
            return instruction.modes
        if len(instruction.modes) == 0:
            return register.modes

        return tuple(int(register.modes[m]) for m in instruction.modes)

    def _apply_to_program_on_register(self, program: "Program", register: Q) -> None:
        for instruction in self.instructions:
            instruction_copy = instruction.copy()

            instruction_copy._apply_to_program_on_register(
                program, register=Q(*self._map_modes(register, instruction))
            )

    def __enter__(self) -> "Program":
        _context.current_program = self

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        _context.current_program = None

    @classmethod
    def from_dict(cls, dict_: dict) -> "Program":
        """Creates a `Program` instance from a `dict`.

        The currently supported format is::

            {
                "instructions": [
                    {
                        "type": <INSTRUCTION_CLASS_NAME>,
                        "attributes": {
                            "constructor_kwargs": <CONSTRUCTOR_KWARGS_IN_DICT_FORMAT>,
                            "modes": <LIST_OF_MODES>
                        }
                    }
                ]
            }

        Args:
            dict_ (dict): The :class:`Program` in a key-value pair format.

        Returns:
            Program: A :class:`Program` initialized using the specified `dict`.
        """

        return cls(
            instructions=[
                Instruction.from_dict(instruction_dict)
                for instruction_dict in dict_["instructions"]
            ]
        )

    def load_blackbird(self, filename: str) -> None:
        """
        Loads the gates to be applied into :attr:`instructions` from a BlackBird file
        (.xbb).

        Args:
            filename (str):
                Location of a Blackbird program (.xbb).
        """
        blackbird_program = blackbird.load(filename)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))

    def loads_blackbird(self, string: str) -> None:
        """
        Loads the gates to apply into :attr:`instructions` from a string
        representing a :class:`~blackbird.program.BlackbirdProgram`.

        Args:
            string (str): String containing a valid Blackbird program.
        """
        blackbird_program = blackbird.loads(string)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))

    def _as_code(self) -> str:
        """Export the :class:`Program` instance as Python code."""

        script = f"with pq.{self.__class__.__name__}() as program:\n"

        four_spaces = " " * 4

        script += "\n".join(
            four_spaces + instruction._as_code() for instruction in self.instructions
        )

        if len(self.instructions) == 0:
            script += four_spaces + "pass"

        return script
