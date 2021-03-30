#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core import _context

from .errors import InvalidModes


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

        if not self._is_distinct(modes):
            raise InvalidModes(
                f"Error registering modes: '{modes}' should be distinct."
            )

        self.modes = modes if modes != (all, ) else tuple()

    def __or__(self, rhs):
        """Registers an `Instruction` or `Program` to the current program.

        If `rhs` is an `Instruction`, then it is appended to the current program's
        `instructions`.

        If `rhs` is a `Program`, then the current program's `instructions` is extended
        with `rhs.instructions`.

        Args:
            rhs (Instruction or Program):
                An `Instruction` or a `Program` to be added to the current program.

        Returns:
            (Q): The current qumode.
        """

        rhs.apply_to_program_on_register(_context.current_program, register=self)

        return self

    __ror__ = __or__

    @staticmethod
    def _is_distinct(iterable):
        return len(iterable) == len(set(iterable))
