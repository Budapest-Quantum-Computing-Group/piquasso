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
        self._instruction_map = self.get_instruction_map()

    @abc.abstractmethod
    def get_instruction_map(self):
        pass

    def execute_instructions(self, instructions):
        """Executes the collected instructions in order.

        Raises:
            NotImplementedError:
                If no such method is implemented on the `Circuit` class.

        Args:
            instructions (list):
                The methods along with keyword arguments of the current circuit to be
                executed in order.
        """
        for instruction in instructions:
            method = self._instruction_map.get(instruction.__class__.__name__)

            if not method:
                raise NotImplementedError(
                    "\n"
                    "No such instruction implemented for this state.\n"
                    "Details:\n"
                    f"instruction={instruction}\n"
                    f"state={self.state}\n"
                    f"Available instructions:\n"
                    + str(", ".join(self._instruction_map.keys())) + "."
                )

            method(instruction)

    def _add_result(self, result):
        if isinstance(result, Iterable):
            self.program.results.extend(result)
        else:
            self.program.results.append(result)
