#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import blackbird

from piquasso.core import _context, _blackbird
from piquasso.core.registry import _create_instance_from_mapping
from piquasso.core.mixins import _RegisterMixin
from .mode import Q


class Program(_RegisterMixin):
    r"""The representation for a quantum program.

    This also specifies a context in which all the instructions should be
    specified.

    Attributes:
        state (State): The initial quantum state.
        circuit (Circuit):
            The circuit on which the quantum program should run.
        instructions (list):
            The set of instructions, e.g. quantum gates and measurements.
    """

    def __init__(
        self,
        state=None,
        instructions=None,
    ):
        self.state = state
        self.instructions = instructions or []

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

        self._circuit = (
            new_state.circuit_class(new_state, program=self)
            if new_state
            else None
        )

    def apply_to_program_on_register(self, program, register):
        if self.state is not None:
            if program.state is not None:
                raise RuntimeError(
                    "The program already has a state registered of type "
                    f"'{type(program.state).__name__}'."
                )

            if register.modes == tuple():
                register.modes = tuple(range(self.state.d))

            self.state.apply_to_program_on_register(program, register)

        for instruction in self.instructions:
            instruction_copy = instruction.copy()

            instruction_copy.apply_to_program_on_register(
                program,
                register=Q(*(register.modes[m] for m in instruction.modes))
            )

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def execute(self):
        """Executes the collected instructions on the circuit."""

        return self._circuit.execute_instructions(self.instructions)

    @property
    def results(self):
        return self._circuit.results

    @classmethod
    def from_properties(cls, properties):
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
            properties (collections.Mapping):
                The desired `Program` instance in the format of a mapping.

        Returns:
            Program: A `Program` initialized using the specified mapping.
        """

        return cls(
            state=_create_instance_from_mapping(properties["state"]),
            instructions=list(
                map(
                    _create_instance_from_mapping,
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
            filename (str): file location of a valid Blackbird program
        """
        blackbird_program = blackbird.load(filename)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))

    def loads_blackbird(self, string):
        """
        Loads the gates to apply into `self.instructions` from a string
        representing a :class:`blackbird.BlackbirdProgram`.

        Args:
            string (str): string containing a valid Blackbird Program
        """
        blackbird_program = blackbird.loads(string)

        self.instructions.extend(_blackbird.load_instructions(blackbird_program))
