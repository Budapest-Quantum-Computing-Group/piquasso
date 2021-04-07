#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
