#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import blackbird

from piquasso import operations
from piquasso.core import _context
from piquasso.core.registry import _create_instance_from_mapping

from . import constants


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
        hbar (float):
            The value of :math:`\hbar` throughout the program, defaults to `2`.
    """

    def __init__(
        self,
        state=None,
        operations=None,
        hbar=constants.HBAR_DEFAULT
    ):
        self._register_state(state)
        self.operations = operations or []
        self.hbar = hbar

    def _register_state(self, state):
        self.state = state

        self.circuit = state._circuit_class(state) if state else None

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
        ```
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
        ```

        TODO: This docstring is quite verbose, put it into a separate Sphinx section
            when present.

        Note:
            Numeric arrays and complex numbers are not supported yet.

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

    def __enter__(self):
        _context.current_program = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _context.current_program = None

    def _blackbird_operation_to_operation(self, blackbird_operation):
        """
        Maps one element of the `operations` of a `BlackbirdProgram` into an
        element of `self.operations`.

        Args:
            blackbird_operation (dict): An element of the `BlackbirdProgram.operations`

        Returns:
            Operation:
                Instance of :class:`Operation` corresponding to the operation defined
                in Blackbird.
        """

        operation_class = {
            "Dgate": operations.D,
            "Xgate": None,
            "Zgate": None,
            "Sgate": operations.S,
            "Pgate": None,
            "Vgate": None,
            "Kgate": None,
            "Rgate": operations.R,
            "BSgate": operations.B,
            "MZgate": None,
            "S2gate": None,
            "CXgate": None,
            "CZgate": None,
            "CKgate": None,
            "Fouriergate": None
        }.get(blackbird_operation["op"])

        operation = operation_class(*blackbird_operation.get("args", tuple()))

        operation.modes = blackbird_operation["modes"]

        return operation

    def from_blackbird(self, bb):
        """
        Loads the gates to apply into `self.operations` from a
        :class:`blackbird.BlackbirdProgram`

        Args:
            bb (blackbird.BlackbirdProgram): the BlackbirdProgram to use
        """
        self.operations = [*map(self._blackbird_operation_to_operation, bb.operations)]

    def load_blackbird(self, filename: str):
        """
        Loads the gates to apply into `self.operations` from a BlackBird file
        (.xbb).

        Args:
            filename (str): file location of a valid Blackbird program
        """
        bb = blackbird.load(filename)
        return self.from_blackbird(bb)

    def loads_blackbird(self, string):
        """
        Loads the gates to apply into `self.operations` from a string
        representing a :class:`blackbird.BlackbirdProgram`.

        Args:
            string (str): string containing a valid Blackbird Program
        """
        bb = blackbird.loads(string)
        return self.from_blackbird(bb)
