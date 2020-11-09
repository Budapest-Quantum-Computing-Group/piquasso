#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of backends."""

from . import registry


class Backend(registry.ClassRecorder):
    def __init__(self, state):
        """
        Args:
            state (State): The initial quantum state.
        """
        self.state = state

    def execute_instructions(self, instructions):
        """Executes the collected instructions in order.

        Args:
            instructions (list): The methods along with keyword arguments of the
                current backend to be executed in order.
        """
        for instruction in instructions:
            operation = instruction["op"]
            operation(self, **instruction["kwargs"])
