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
from piquasso.api.errors import InvalidState, PiquassoException
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

        return np.outer(state_vector, state_vector.conj())

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        if len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

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

    def quadratures_mean_variance(
        self, modes: Tuple[int, ...], phi: float = 0
    ) -> Tuple[float, float]:
        r"""This method calculates the mean and the variance of the qudrature operators
        for a single qumode state.
        The quadrature operators :math:`x` and :math:`p` for a mode :math:`i`
        can be calculated using the creation and annihilation operators as follows:

        .. math::
            x_i &= \sqrt{\frac{\hbar}{2}} (a_i + a_i^\dagger) \\
                p_i &= -i \sqrt{\frac{\hbar}{2}} (a_i - a_i^\dagger).

        Let :math:`\phi \in [ 0, 2 \pi )`, we can rotate the quadratures using the
        following transformation:

        .. math::
            Q_{i, \phi} = \cos\phi~x_i + \sin\phi~p_i.

        The expectation value :math:`\langle Q_{i, \phi}\rangle` can be calculated as:

        .. math::
            \operatorname{Tr}(\rho_i Q_{i, \phi}),

        where :math:`\rho_i` is the reduced density matrix of the mode :math:`i` and
        :math:`Q_{i, \phi}` is the rotated quadrature operator for a single mode and
        the variance is calculated as:

        .. math::
            \operatorname{\textit{Var}(Q_{i,\phi})} = \langle Q_{i, \phi}^{2}\rangle
                - \langle Q_{i, \phi}\rangle^{2}.

        Args:
            phi (float): The rotation angle. By default it is `0` which means that
                the mean of the position operator is being calculated. For :math:`\phi=
                \frac{\pi}{2}` the mean of the momentum operator is being calculated.
            modes (tuple[int]): The correspoding mode at which the mean of the
                quadratures are being calculated.
        Returns:
            (float, float): A tuple that contains the expectation value and the
                varianceof of the quadrature operator respectively.
        """
        reduced_dm = self.reduced(modes=modes).density_matrix
        annih = np.diag(np.sqrt(np.arange(1, self._config.cutoff)), 1)
        create = annih.T
        position = (create + annih) * np.sqrt(self._config.hbar / 2)
        momentum = -1j * (annih - create) * np.sqrt(self._config.hbar / 2)

        if phi != 0:
            rotated_quadratures = position * np.cos(phi) + momentum * np.sin(phi)
        else:
            rotated_quadratures = position

        expctation = np.trace(np.dot(reduced_dm, rotated_quadratures)).real
        variance = (
            np.trace(
                np.dot(reduced_dm, np.dot(rotated_quadratures, rotated_quadratures))
            ).real
            - expctation ** 2
        )
        return expctation, variance
