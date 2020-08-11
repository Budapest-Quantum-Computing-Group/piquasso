#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.context import Context


class Program:
    """The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.
    """

    def __init__(self, state, backend_class=None):
        """
        Args:
            state (State): The initial quantum state.
            backend_class: The backend on which the quantum program should run.
        """
        self.state = state
        self.d = self.state.d
        self.instructions = []

        backend_class = backend_class or state.backend_class
        self.backend = backend_class(state)

    def execute(self):
        """Executes the collected instructions on the backend."""

        self.backend.execute_instructions(self.instructions)

    def __enter__(self):
        Context.current_program = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Context.current_program = None
