#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import copy

from piquasso.core import _context

from .errors import InvalidModes
from .program import Program
from .state import State


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

        self.modes = modes

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
        if isinstance(rhs, Program):
            self._register_program(rhs)
        elif isinstance(rhs, State):
            self._register_state(rhs)
        else:
            rhs.modes = self.modes
            _context.current_program.instructions.append(rhs)

        return self

    __ror__ = __or__

    def _register_program(self, program):
        if program.state is not None:
            if _context.current_program.state is not None:
                raise RuntimeError(
                    "The current program already has a state registered of type "
                    f"'{type(_context.current_program.state).__name__}'."
                )

            self._register_state(program.state)

        _context.current_program.instructions += map(
            self._resolve_instruction, program.instructions
        )

    def _register_state(self, state):
        state_copy = copy.deepcopy(state)
        _context.current_program._register_state(state_copy)

    def _resolve_instruction(self, instruction):
        new_instruction = copy.deepcopy(instruction)
        new_instruction.modes = (
            None if instruction.modes is None
            else tuple(self.modes[m] for m in instruction.modes)
        )
        return new_instruction

    @staticmethod
    def _is_distinct(iterable):
        return len(iterable) == len(set(iterable))
