#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import copy
import blackbird

from piquasso.core import _context, _blackbird
from piquasso.core.registry import _create_instance_from_mapping


class Program:
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
        self._register_state(state)
        self.instructions = instructions or []
        self.results = []

    def _register_state(self, state):
        self.state = state

        self.circuit = state.circuit_class(state, program=self) if state else None

    def copy(self):
        return copy.deepcopy(self)

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def execute(self):
        """Executes the collected instructions on the circuit."""

        self.circuit.execute_instructions(self.instructions)

        return self.results

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
