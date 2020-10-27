#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.context import Context


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

    def __or__(self, op):
        """This registers the specified `operator` to the current quantum
            program.

        Args:
            op (Operation): The operator to be applied on execution.

        Returns:
            (Q): The current qumode.
        """
        Context.current_program.instructions.append(
            {
                'params': op.params,
                'modes': self.modes,
                'op': op.resolve_method_for_backend()
            }
        )
        return self

    @staticmethod
    def _is_distinct(iterable):
        return len(iterable) == len(set(iterable))
