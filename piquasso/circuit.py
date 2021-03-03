#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of circuits."""

import abc


class Circuit(abc.ABC):
    _operation_map = {}

    def __init__(self, state):
        """
        Args:
            state (State): The initial quantum state.
        """
        self.state = state
        self._operation_map.update(self.get_operation_map())

    @abc.abstractmethod
    def get_operation_map(self):
        pass

    @classmethod
    def execute_operations(cls, operations):
        """Executes the collected operations in order.

        Raises:
            NotImplementedError:
                If no such method is implemented on the `Circuit` class.

        Args:
            operations (list):
                The methods along with keyword arguments of the current circuit to be
                executed in order.
        """
        for operation in operations:
            method = cls._operation_map.get(operation.__class__.__name__)

            if not method:
                raise NotImplementedError(
                    "No such operation implemented on this circuit."
                )

            method(operation)
