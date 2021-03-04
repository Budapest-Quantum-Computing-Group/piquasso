#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import copy

from piquasso import context
from piquasso import Program


class Q:
    """
    The implementation of qumodes, which is used to track on which qumodes are
    the operators placed in the circuit.
    """

    def __init__(self, *modes):
        """
        Args:
            modes: Distinct positive integer values which are used to represent
                qumodes.
        """

        assert self._is_distinct(modes)

        self.modes = modes

    def __or__(self, rhs):
        """Registers an `Operation` or `Program` to the current program.

        If `rhs` is an `Operation`, then it is appended to the current program's
        `operations`.

        If `rhs` is a `Program`, then the current program's `operations` is extended
        with `rhs.operations`.

        Args:
            rhs (Operation or Program): An `Operation` or a `Program` to be added to the
                current program.

        Returns:
            (Q): The current qumode.
        """
        if isinstance(rhs, Program):
            self._register_program(rhs)
        else:
            rhs.modes = self.modes
            context.current_program.operations.append(rhs)

        return self

    def _register_program(self, program):
        context.current_program.operations += \
            map(self._resolve_operation, program.operations)

    def _resolve_operation(self, operation):
        new_operation = copy.deepcopy(operation)
        new_operation.modes = (
            None if operation.modes is None
            else tuple(self.modes[m] for m in operation.modes)
        )
        return new_operation

    @staticmethod
    def _is_distinct(iterable):
        return len(iterable) == len(set(iterable))
