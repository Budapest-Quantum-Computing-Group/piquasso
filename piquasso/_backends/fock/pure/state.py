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

from typing import Tuple, Generator, Any, Dict

import numpy as np

from piquasso.api.config import Config
from piquasso.api.errors import InvalidState
from piquasso._math.fock import cutoff_cardinality, FockBasis

from ..state import BaseFockState
from ..general.state import FockState


class PureFockState(BaseFockState):
    r"""A simulated pure Fock state.

    If no mixed states are needed for a Fock simulation, then this state is the most
    appropriate currently, since it does not store an entire density matrix, only a
    state vector with size

    .. math::
        {d + c - 1 \choose c - 1},

    where :math:`c \in \mathbb{N}` is the Fock space cutoff.
    """

    def __init__(self, *, d: int, config: Config = None) -> None:
        """
        Args:
            state_vector (numpy.ndarray, optional): The initial state vector.
            d (int): The number of modes.
            cutoff (int): The Fock space cutoff.
        """

        super().__init__(d=d, config=config)

        self._state_vector = self._get_empty()

    def _get_empty(self) -> np.ndarray:
        return np.zeros(shape=(self._space.cardinality,), dtype=complex)

    def reset(self) -> None:
        self._state_vector = self._get_empty()
        self._state_vector[0] = 1.0

    @property
    def nonzero_elements(self) -> Generator[Tuple[complex, FockBasis], Any, None]:
        for index, basis in self._space.basis:
            coefficient: complex = self._state_vector[index]
            if coefficient != 0:
                yield coefficient, basis

    def __repr__(self) -> str:
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PureFockState):
            return False
        return np.allclose(self._state_vector, other._state_vector)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self.d, cutoff=self._config.cutoff)

        state_vector = self._state_vector[:cardinality]

        return np.outer(state_vector, state_vector)

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        index = self._space.index(occupation_number)

        return np.real(
            self._state_vector[index].conjugate() * self._state_vector[index]
        )

    @property
    def fock_probabilities(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self._space.d, cutoff=self._config.cutoff)

        return (self._state_vector * self._state_vector.conjugate()).real[:cardinality]

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        for index, basis in self._space.basis:
            probability_map[tuple(basis)] = np.abs(self._state_vector[index]) ** 2

        return probability_map

    def normalize(self) -> None:
        if np.isclose(self.norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._state_vector = self._state_vector / np.sqrt(self.norm)

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the norm of the state vector is not close to 1.0.
        """
        sum_of_probabilities = sum(self.fock_probabilities)

        if not np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )
