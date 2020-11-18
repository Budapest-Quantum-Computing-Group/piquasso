#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import json
import blackbird

from piquasso import constants, registry
from piquasso.context import Context
from piquasso.operations import Operation


class Program:
    """The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.

    Attributes:
        state (State): The initial quantum state.
        backend (Backend):
            The backend on which the quantum program should run.
        operations (list):
            The set of operations, e.g. quantum gates and measurements.
        hbar (float):
            The value of :math:`\hbar` throughout the program, defaults to `2`.
    """

    def __init__(
        self,
        state=None,
        backend_class=None,
        operations=None,
        hbar=constants.HBAR_DEFAULT
    ):
        self.state = state
        self.operations = operations or []
        self.hbar = hbar

        if backend_class is not None:
            self.backend = backend_class(state)

        elif state is not None:
            self.backend = state.backend_class(state)

        else:
            self.backend = None

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
            "backend_class": <BACKEND_CLASS_NAME>,
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
            state=registry.create_instance_from_mapping(properties["state"]),
            backend_class=registry.retrieve_class(properties["backend_class"]),
            operations=list(
                map(registry.create_instance_from_mapping, properties["operations"])
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
        """Executes the collected operations on the backend."""

        self.backend.execute_operations(self.operations)

    def __enter__(self):
        Context.current_program = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Context.current_program = None

    def _blackbird_operation_to_operation(self, blackbird_operation):
        """
        Maps one element of the `operations` of a `BlackbirdProgram` into an
        element of `self.operations`.

        Args:
            operation (dict): An element of the `BlackbirdProgram.operations`
        """

        operation_class = Operation.blackbird_op_to_gate(blackbird_operation["op"])

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
        self.operations = \
            [*map(self._blackbird_operation_to_operation, bb.operations)]

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
