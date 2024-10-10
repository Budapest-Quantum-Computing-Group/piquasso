#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from typing import Optional, Tuple, Dict

import numpy as np

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso.api.connector import BaseConnector

from piquasso._math.fock import cutoff_fock_space_dim, get_fock_space_basis
from piquasso._math.linalg import vector_absolute_square
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_space_array,
    get_auxiliary_modes,
)

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

    :ivar state_vector: The state vector of the quantum state.
    """

    def __init__(
        self, *, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            connector (BaseConnector): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """

        super().__init__(d=d, connector=connector, config=config)

        self.state_vector = self._get_empty()

    def _get_empty_list(self) -> list:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)
        return [0.0] * state_vector_size

    def _get_empty(self) -> np.ndarray:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)

        return self._np.zeros(
            shape=(state_vector_size,), dtype=self._config.complex_dtype
        )

    def reset(self) -> None:
        state_vector_list = self._get_empty_list()
        state_vector_list[0] = 1.0

        self.state_vector = self._np.array(
            state_vector_list, dtype=self._config.complex_dtype
        )

    def _nonzero_elements_for_single_state_vector(self, state_vector):
        np = self._connector.np
        nonzero_indices = np.nonzero(state_vector)[0]

        occupation_numbers = self._space[nonzero_indices]

        nonzero_elements = state_vector[nonzero_indices]

        for index, coefficient in enumerate(nonzero_elements):
            yield coefficient, tuple(occupation_numbers[index])

    @property
    def nonzero_elements(self):
        return self._nonzero_elements_for_single_state_vector(self.state_vector)

    def _get_repr_for_single_state_vector(self, nonzero_elements):
        return " + ".join(
            [str(coefficient) + str(basis) for coefficient, basis in nonzero_elements]
        )

    def __str__(self) -> str:
        return self._get_repr_for_single_state_vector(self.nonzero_elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PureFockState):
            return False
        return self._np.allclose(self.state_vector, other.state_vector)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff)

        state_vector = self.state_vector[:cardinality]

        return self._np.outer(state_vector, self._np.conj(state_vector))

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        index = get_index_in_fock_space(occupation_number)

        return self._np.real(
            self.state_vector[index].conjugate() * self.state_vector[index]
        )

    def get_particle_detection_probability_on_modes(
        self,
        occupation_numbers: np.ndarray,
        modes: Tuple[int, ...],
    ) -> float:
        np = self._connector.np
        fallback_np = self._connector.fallback_np

        occupation_numbers = fallback_np.array(occupation_numbers)

        auxiliary_d = self.d - len(modes)
        auxiliary_cutoff = self._config.cutoff - sum(occupation_numbers)
        auxiliary_modes = get_auxiliary_modes(self.d, modes)

        unordered_modes = fallback_np.concatenate([modes, auxiliary_modes])

        card = cutoff_fock_space_dim(d=auxiliary_d, cutoff=auxiliary_cutoff)
        auxiliary_basis = get_fock_space_basis(d=auxiliary_d, cutoff=auxiliary_cutoff)

        repeated_occupation_numbers = fallback_np.repeat(
            occupation_numbers[None, :], card, axis=0
        )

        unordered_occupation_numbers = fallback_np.concatenate(
            [repeated_occupation_numbers, auxiliary_basis], axis=1
        )

        sorter = fallback_np.argsort(unordered_modes)

        ordered_occupation_numbers = unordered_occupation_numbers[:, sorter]

        indices = get_index_in_fock_space_array(ordered_occupation_numbers)

        return np.sum(np.abs(self.state_vector[indices]) ** 2)

    @property
    def fock_probabilities(self) -> np.ndarray:
        return vector_absolute_square(self.state_vector, self._connector)

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        for index, basis in enumerate(self._space):
            probability_map[tuple(basis)] = self._np.abs(self.state_vector[index]) ** 2

        return probability_map

    def normalize(self) -> None:
        norm = self.norm

        if self._config.validate and np.isclose(norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self.state_vector = self.state_vector / self._np.sqrt(norm)

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the norm of the state vector is not close to 1.0.
        """
        if not self._config.validate:
            return

        sum_of_probabilities = sum(self.fock_probabilities)

        if not self._np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )

    def _get_mean_position_indices(self, mode):
        fallback_np = self._connector.fallback_np

        self._space[:, mode] -= 1
        lowered_indices = get_index_in_fock_space_array(self._space)
        self._space[:, mode] += 2
        raised_indices = get_index_in_fock_space_array(self._space)
        self._space[:, mode] -= 1

        relevant_column = self._space[:, mode]

        nonzero_indices_on_mode = (relevant_column > 0).nonzero()[0]
        upper_index = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff - 1)

        multipliers = fallback_np.sqrt(
            fallback_np.concatenate(
                [
                    relevant_column[nonzero_indices_on_mode],
                    relevant_column[:upper_index] + 1,
                ]
            )
        )
        left_indices = fallback_np.concatenate(
            [lowered_indices[nonzero_indices_on_mode], raised_indices[:upper_index]]
        )
        right_indices = fallback_np.concatenate(
            [nonzero_indices_on_mode, fallback_np.arange(upper_index)]
        )

        return multipliers, left_indices, right_indices

    def mean_position(self, mode: int) -> np.ndarray:
        np = self._connector.np
        fallback_np = self._connector.fallback_np
        multipliers, left_indices, right_indices = self._get_mean_position_indices(mode)

        state_vector = self.state_vector

        accumulator = np.dot(
            (multipliers * state_vector[left_indices]),
            state_vector[right_indices],
        )

        return np.real(accumulator) * fallback_np.sqrt(self._config.hbar / 2)

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
        np = self._connector.np

        reduced_dm = self.reduced(modes=modes).density_matrix
        annih = np.diag(np.sqrt(np.arange(1, self._config.cutoff)), 1)
        create = annih.T
        position = (create + annih) * np.sqrt(self._config.hbar / 2)
        momentum = -1j * (annih - create) * np.sqrt(self._config.hbar / 2)

        if phi != 0:
            rotated_quadratures = position * np.cos(phi) + momentum * np.sin(phi)
        else:
            rotated_quadratures = position

        expectation = np.real(np.trace(np.dot(reduced_dm, rotated_quadratures)))
        variance = (
            np.real(
                np.trace(
                    np.dot(reduced_dm, np.dot(rotated_quadratures, rotated_quadratures))
                )
            )
            - expectation**2
        )
        return expectation, variance

    def mean_photon_number(self):
        r"""Returns the mean photon number

        .. math::
            \mathbb{E}_{\ket{\psi}}(\hat{n}) :=
                \bra{\psi} \hat{n} \ket{\psi},

        where :math:`\hat{n}` is the total photon number operator.
        """

        numbers = np.sum(self._space, axis=1)

        return numbers @ (self._np.abs(self.state_vector) ** 2)

    def variance_photon_number(self):
        r"""Returns the photon number variance

        .. math::
            \operatorname{Var}_{\ket{\psi}}(\hat{n}) :=
                \bra{\psi} (\hat{n} - \bar{n})^2 \ket{\psi},

        where :math:`\hat{n}` is the total photon number operator and :math:`\bar{n}`
        is its expectation value given by :meth:`mean_photon_number`.
        """

        numbers = np.sum(self._space, axis=1)

        probabilities = self._np.abs(self.state_vector) ** 2

        mean = numbers @ probabilities

        return (numbers - mean) ** 2 @ probabilities

    def get_tensor_representation(self):
        cutoff = self._config.cutoff
        d = self.d

        return self._connector.scatter(
            self._space,
            self.state_vector,
            [cutoff] * d,
        )

    def copy(self) -> "PureFockState":
        # NOTE: `__deepcopy__` is not allowed for tensorflow variables, so we have to
        # do it explicitely here.
        state = self.__class__(
            d=self.d, connector=self._connector, config=self._config.copy()
        )

        state.state_vector = self._connector.np.copy(self.state_vector)

        return state

    def get_purity(self):
        return 1.0
