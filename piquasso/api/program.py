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

import json
import blackbird

from piquasso.api.errors import InvalidProgram, InvalidParameter
from piquasso.core import _context, _blackbird, _registry
from piquasso.core import _mixins
from .mode import Q


class Program(_mixins.RegisterMixin):
    r"""The class representing the quantum program.

    A `Program` object can be used with the `with` statement. In this context, the state
    and all the instructions could be specified.

    A simple example usage:

    .. code-block:: python

        import numpy as np
        import piquasso as pq

        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=5) | pq.Vacuum()

            pq.Q(0, 1) | pq.Squeezing(r=0.5)

        results = program.execute()

    Args:
        state (State): The initial quantum state.
        instructions (list[~piquasso.api.instruction.Instruction], optional):
            The set of instructions, e.g. quantum gates and measurements.
    """

    def __init__(
        self,
        state=None,
        instructions: list = None,
    ):
        self.state = state
        self.instructions = instructions or []

    @property
    def state(self):
        """The quantum state corresponding to the program."""

        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

        self._circuit = (
            new_state.circuit_class(program=self)
            if new_state
            else None
        )

    def _apply_to_program_on_register(self, program, register):
        if self.state is not None:
            if program.state is not None:
                raise InvalidProgram(
                    "The program already has a state registered of type "
                    f"'{type(program.state).__name__}'."
                )

            if register.modes == tuple():
                register.modes = tuple(range(self.state.d))

            self.state._apply_to_program_on_register(program, register)

        for instruction in self.instructions:
            instruction_copy = instruction.copy()

            instruction_copy._apply_to_program_on_register(
                program,
                register=Q(*(register.modes[m] for m in instruction.modes))
            )

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def execute(self, shots=1) -> list:
        """Executes the collected instructions on the circuit.

        Args:
            shots (int): The number of samples to generate.

        Returns:
            list[Result]: A list of the execution results.
        """

        if not isinstance(shots, int) or shots < 1:
            raise InvalidParameter(
                f"The number of shots should be a positive integer: shots={shots}."
            )

        return self._circuit.execute_instructions(
            self.instructions,
            self.state,
            shots=shots,
        )

    @property
    def results(self) -> list:
        return self._circuit.results

    @classmethod
    def from_properties(cls, properties: dict):
        """Creates a `Program` instance from a mapping.

        The currently supported format is

        .. code-block:: python

            {
                "state": {
                    "type": <STATE_CLASS_NAME>,
                    "properties": {
                        ...
                    }
                },
                "instructions": [
                    {
                        "type": <INSTRUCTION_CLASS_NAME>,
                        "properties": {
                            ...
                        }
                    }
                ]
            }

        Note:
            Numeric arrays and complex numbers are not yet supported.

        Args:
            properties (dict):
                The desired `Program` instance in the format of a mapping.

        Returns:
            Program: A `Program` initialized using the specified mapping.
        """

        return cls(
            state=_registry.create_instance_from_mapping(properties["state"]),
            instructions=list(
                map(
                    _registry.create_instance_from_mapping,
                    properties["instructions"],
                )
            )
        )

    @classmethod
    def from_json(cls, json_):
        """Creates a `Program` instance from JSON.

        Almost the same as :meth:`from_properties`, but with JSON parsing.

        Args:
            json_ (str): The JSON formatted program.

        Returns:
            Program: A program initialized with the JSON data.
        """
        properties = json.loads(json_)

        return cls.from_properties(properties)

    def load_blackbird(self, filename: str):
        """
        Loads the gates to apply into `self.instructions` from a BlackBird file
        (.xbb).

        Args:
            filename (str):
                Location of a Blackbird program (.xbb).
        """
        blackbird_program = blackbird.load(filename)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))

    def loads_blackbird(self, string: str):
        """
        Loads the gates to apply into `self.instructions` from a string
        representing a :class:`~blackbird.program.BlackbirdProgram`.

        Args:
            string (str):
                String containing a valid Blackbird program.
        """
        blackbird_program = blackbird.loads(string)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))

    def as_code(self):
        """Export the :class:`Program` instance as Python code."""

        with_statement = f"with pq.{self.__class__.__name__}() as program:"

        script = (
            f"import piquasso as pq\n\n\n{with_statement}\n"
        )

        four_space = " " * 4

        if self.state is not None:
            script += four_space + self.state._as_code() + "\n"

        script += "\n"

        for instruction in self.instructions:
            script += four_space + instruction._as_code() + "\n"

        return script
