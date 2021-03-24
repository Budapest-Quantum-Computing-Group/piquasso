#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import blackbird

from piquasso.core import _context, _blackbird
from piquasso.core.registry import _create_instance_from_mapping


class Program:
    r"""The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.

    Attributes:
        state (State): The initial quantum state.
        circuit (Circuit):
            The circuit on which the quantum program should run.
        operations (list):
            The set of operations, e.g. quantum gates and measurements.
    """

    def __init__(
        self,
        state=None,
        operations=None,
    ):
        self._register_state(state)
        self.operations = operations or []
        self.results = []

    def _register_state(self, state):
        self.state = state

        self.circuit = state.circuit_class(state, program=self) if state else None

    @property
    def d(self):
        """The number of modes, on which the state of the program is defined.

        Returns:
            int: The number of modes.
        """
        return self.state.d

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
                "operations": [
                    {
                        "type": <OPERATION_CLASS_NAME>,
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
            operations=list(
                map(
                    _create_instance_from_mapping,
                    properties["operations"],
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

    def execute(self):
        """Executes the collected operations on the circuit."""

        self.circuit.execute_operations(self.operations)

        return self.results

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def load_blackbird(self, filename: str):
        """
        Loads the gates to apply into `self.operations` from a BlackBird file
        (.xbb).

        Args:
            filename (str): file location of a valid Blackbird program
        """
        blackbird_program = blackbird.load(filename)

        self.operations.extend(_blackbird.load_operations(blackbird_program))

    def loads_blackbird(self, string):
        """
        Loads the gates to apply into `self.operations` from a string
        representing a :class:`blackbird.BlackbirdProgram`.

        Args:
            string (str): string containing a valid Blackbird Program
        """
        blackbird_program = blackbird.loads(string)

        self.operations.extend(_blackbird.load_operations(blackbird_program))
