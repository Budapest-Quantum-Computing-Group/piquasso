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

    def execute_operations(self, operations):
        """Executes the collected operations in order.

        Args:
            operations (list): The methods along with keyword arguments of the
                current backend to be executed in order.
        """
        for operation in operations:
            operation(self)
