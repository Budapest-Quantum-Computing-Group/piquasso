#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.context import Context


class Program:
    """The representation for a quantum program.

    This also specifies a context in which all the operations should be
    specified.
    """

    def __init__(self, state, backend):
        """
        Args:
            state (State): The initial quantum state.
            backend: The backend on which the quantum program should run.
        """
        self.state = state
        self.d = self.state.d
        self.instructions = []
        self.backend = backend(state)

    def execute(self):
        """Execute all the collected instructions in order.

        TODO: Multiple particles are not handled yet.
        """

        self.backend.execute_program(self)

    def __enter__(self):
        Context.current_program = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Context.current_program = None
