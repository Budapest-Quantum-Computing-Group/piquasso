#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of circuits."""

import abc

from collections.abc import Iterable


class Circuit(abc.ABC):
    def __init__(self, state, program):
        """
        Args:
            state (State): The initial quantum state.
        """
        self.state = state
        self.program = program
        self._operation_map = self.get_operation_map()

    @abc.abstractmethod
    def get_operation_map(self):
        pass

    def execute_operations(self, operations):
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
            method = self._operation_map.get(operation.__class__.__name__)

            if not method:
                raise NotImplementedError(
                    "\n"
                    "No such operation implemented for this state.\n"
                    "Details:\n"
                    f"operation={operation}\n"
                    f"state={self.state}\n"
                    f"Available operations:\n"
                    + str(", ".join(self._operation_map.keys())) + "."
                )

            method(operation)

    def _add_result(self, result):
        if isinstance(result, Iterable):
            self.program.results.extend(result)
        else:
            self.program.results.append(result)
