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

    def __init__(self, state, backend_class=None, hbar=constants.HBAR_DEFAULT):
        """
        Args:
            state (State): The initial quantum state.
            backend_class: The backend on which the quantum program should run.
        """
        self.state = state
        self.d = self.state.d
        self.instructions = []
        self.hbar = hbar

        self.backend = self._create_backend(backend_class)

    def execute(self):
        """Executes the collected instructions on the backend."""

        self.backend.execute_instructions(self.instructions)

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

    def _blackbird_operation_to_instruction(self, blackbird_operation):
        """
        Maps one element of the `operations` of a `BlackbirdProgram` into an
        element of `self.instructions`.

        Args:
            operation (dict): An element of the `BlackbirdProgram.operations`
        """

        operation_class = Operation.blackbird_op_to_gate(blackbird_operation["op"])
        instruction = {
            "kwargs": {
                "modes": blackbird_operation["modes"],
            },
            "op": operation_class.backends[self.backend.__class__]
        }

        if "args" in blackbird_operation:
            instruction["kwargs"]["params"] = blackbird_operation["args"]

        return instruction

    def from_blackbird(self, bb):
        """
        Loads the gates to apply into `self.instructions` from a
        :class:`blackbird.BlackbirdProgram`

        Args:
            bb (blackbird.BlackbirdProgram): the BlackbirdProgram to use
        """
        self.instructions = \
            [*map(self._blackbird_operation_to_instruction, bb.operations)]

    def load_blackbird(self, filename: str):
        """
        Loads the gates to apply into `self.instructions` from a BlackBird file
        (.xbb).

        Args:
            filename (str): file location of a valid Blackbird program
        """
        bb = blackbird.load(filename)
        return self.from_blackbird(bb)

    def loads_blackbird(self, string):
        """
        Loads the gates to apply into `self.instructions` from a string
        representing a :class:`blackbird.BlackbirdProgram`.

        Args:
            string (str): string containing a valid Blackbird Program
        """
        bb = blackbird.loads(string)
        return self.from_blackbird(bb)
