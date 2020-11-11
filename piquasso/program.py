#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import blackbird

from piquasso import constants, registry
from piquasso.context import Context
from piquasso.operations import Operation


class Program:
    """The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.
    """

    def __init__(self, state=None, backend_class=None, hbar=constants.HBAR_DEFAULT):
        """
        Args:
            state (State): The initial quantum state.
            backend_class: The backend on which the quantum program should run.
        """
        self.state = state
        self.operations = []
        self.hbar = hbar

        self.backend = (
            None if (backend_class or self.state) is None
            else self._create_backend(backend_class)
        )

    @property
    def d(self):
        """The number of modes, on which the state of the program is defined.

        Returns:
            int: The number of modes.
        """
        return self.state.d

    def execute(self):
        """Executes the collected operations on the backend."""

        self.backend.execute_operations(self.operations)

    def __enter__(self):
        Context.current_program = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Context.current_program = None

    def _create_backend(self, backend_class):
        """Instantiates a backend from the specified `backend_class`.

        Args:
            backend_class: The class (or its name) to be instantiated.

        Returns:
            Backend: The instantiated backend.
        """
        backend_class = backend_class or self.state.backend_class

        if isinstance(backend_class, str):
            backend_class = registry.retrieve_class(backend_class)

        return backend_class(self.state)

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
