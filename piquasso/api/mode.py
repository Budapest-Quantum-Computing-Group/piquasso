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

import typing
from typing import Collection, Tuple, Union, Any

from piquasso.core import _context

from .errors import InvalidModes, PiquassoException

if typing.TYPE_CHECKING:
    from piquasso.api.instruction import Instruction
    from piquasso.api.program import Program


class Q:
    """
    The implementation of qumodes, which is used to track on which qumodes are
    the operators placed in the circuit.

    Example usage::

        import numpy as np
        import piquasso as pq

        with pq.Program() as program:
            pq.Q(0, 1) | pq.Squeezing(r=0.5)

        simulator = pq.GaussianSimulator(d=5)
        result = simulator.execute(program)

    In the above example, the :class:`~piquasso.instructions.gates.Squeezing` gate
    is applied to modes `0, 1`.

    Note, that it is not necessarily required to specify any modes, if the context
    permits, e.g. when registering states.

    One could use the `all` keyword to indicate that the registered
    :class:`~piquasso.api.instruction.Instruction` mashould be applied to all modes,
    i.e.::

        with pq.Program() as program:
            pq.Q(0, 1) | pq.Squeezing(r=0.5)

            pq.Q(all) | pq.ParticleNumberMeasurement()

    or alternatively no parameters, i.e.::

        with pq.Program() as program:
            pq.Q(0, 1) | pq.Squeezing(r=0.5)

            pq.Q() | pq.ParticleNumberMeasurement()
    """

    def __init__(self, *modes: Union[int, Any]) -> None:
        """
        Args:
            modes (int):
                Variable length list of non-negative integers specifying the modes.

        Raises:
            InvalidModes:
                Raised if
                - the specified modes are not distinct;
                - negative integers were specified.
        """
        is_all = modes == (all,)

        if not is_all and any(mode < 0 for mode in modes):
            raise InvalidModes(
                f"Error registering modes: '{modes}' should be non-negative."
            )

        if not is_all and not self._is_distinct(modes):
            raise InvalidModes(
                f"Error registering modes: '{modes}' should be distinct."
            )

        self.modes: Tuple[int, ...] = modes if not is_all else tuple()

    def __or__(self, rhs: Union["Instruction", "Program"]) -> "Q":
        """Registers an `Instruction` or `Program` to the current program.

        If `rhs` is an `Instruction`, then it is appended to the current program's
        `instructions`.

        If `rhs` is a `Program`, then the current program's `instructions` is extended
        with the `instructions` of `rhs`.

        Args:
            rhs (Instruction or Program):
                An `Instruction` or a `Program` to be added to the current program.

        Returns:
            (Q): The current qumode.
        """

        if _context.current_program is None:
            raise PiquassoException(
                "The current program is None; therefore, it is not possible to register"
                " the parameters on the given modes.\n"
                "Please use `Q(...) | ...` only inside `with pq.Program() as program:` "
                "blocks, e.g.\n"
                "\n"
                "with pq.Program() as program:\n"
                "    pq.Q(0, 1) | pq.Squeezing(r=0.5)"
            )

        rhs._apply_to_program_on_register(_context.current_program, register=self)

        return self

    __ror__ = __or__

    @staticmethod
    def _is_distinct(iterable: Collection) -> bool:
        return len(iterable) == len(set(iterable))
