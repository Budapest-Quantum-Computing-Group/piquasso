#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.context import Context


class Q:
    """
    The implementation of qumodes, which is used to track on which qumodes are the
    operators placed in the circuit.
    """
    def __init__(self, *modes):
        """
        Args:
            modes: The positive integer values which is used to represent the qumodes.
        """
        self.modes = modes

    def __or__(self, gate):
        """This registers the specified `operator` to the current quantum program.

        Args:
            gate (Gate): The operator to be applied on execution.

        Returns:
            (Q): The current qumode.
        """
        Context.current_program.instructions.append(
            {'params': gate.params, 'modes': self.modes, 'op': gate.resolve_method_for_backend()}
        )
        return self
