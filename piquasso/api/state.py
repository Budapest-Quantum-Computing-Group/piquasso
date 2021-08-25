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

import abc
import copy
from typing import Tuple, Optional, Type

import numpy as np
import numpy.typing as npt

from piquasso.api.circuit import Circuit
from piquasso.api.program import Program
from piquasso.api.errors import InvalidParameter
from piquasso.api.result import Result

from piquasso.core import _mixins, _registry


class State(_mixins.DictMixin, _mixins.CodeMixin, abc.ABC):
    """The base class from which all `*State` classes are derived.

    Properties:
        circuit_class (~piquasso.api.circuit.Circuit):
            Class attribute for specifying corresponding circuit. The circuit is
            responsible to execute the specified instructions on the :class:`State`
            instance.
        d (int): Instance attribute specifying the number of modes.
    """

    @property
    @abc.abstractmethod
    def circuit_class(self) -> Type[Circuit]:
        pass

    @property
    @abc.abstractmethod
    def d(self) -> int:
        pass

    @classmethod
    def from_dict(cls, dict_: dict) -> "State":
        class_ = _registry.get_class(dict_["type"])
        return class_(**dict_["attributes"]["constructor_kwargs"])

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def _as_code(self) -> str:
        return f"pq.Q() | pq.{self.__class__.__name__}(d={self.d})"

    def apply(self, program: Program, shots: int = 1) -> Optional[Result]:
        """Applyes the given program to the state and executes it.

        Args:
            program (Program):
                The program whose instructions are used in the simpulation.
            shots (int):
                The number of samples to generate.
        """
        if not isinstance(shots, int) or shots < 1:
            raise InvalidParameter(
                f"The number of shots should be a positive integer: shots={shots}."
            )

        circuit = self.circuit_class()
        return circuit.execute_instructions(
            program.instructions,
            self,
            shots=shots,
        )

    @staticmethod
    def _get_operator_index(
        modes: Tuple[int, ...]
    ) -> Tuple[npt.NDArray[np.intc], npt.NDArray[np.intc]]:
        """
        Note:
            For indexing of numpy arrays, see
            https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
        """

        transformed_columns = np.array([modes] * len(modes))
        transformed_rows = transformed_columns.transpose()

        return transformed_rows, transformed_columns

    def _get_auxiliary_modes(self, modes: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(np.delete(np.arange(self.d), modes))

    @staticmethod
    def _get_auxiliary_operator_index(
        modes: Tuple[int, ...],
        auxiliary_modes: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        auxiliary_rows = tuple(np.array([modes] * len(auxiliary_modes)).transpose())

        return auxiliary_rows, auxiliary_modes

    @abc.abstractmethod
    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        """
        Returns the particle number detection probability using the occupation number
        specified as a parameter.

        Args:
            occupation_number (tuple):
                List of natural numbers representing the number of particles in each
                mode.

        Returns:
            float: The probability of detection.
        """
        pass
