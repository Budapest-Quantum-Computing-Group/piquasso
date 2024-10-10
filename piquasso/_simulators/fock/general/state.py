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

from typing import Optional, Tuple, Any, Generator, Dict

import numpy as np
from piquasso.api.connector import BaseConnector

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso._math.linalg import is_selfadjoint
from piquasso._math.fock import cutoff_fock_space_dim, get_fock_space_basis
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_space_array,
)

from ..state import BaseFockState


class FockState(BaseFockState):
    """Object to represent a general bosonic state in the Fock basis.

    Note:
        If you only work with pure states, it is advised to use
        :class:`~piquasso._simulators.fock.pure.state.PureFockState` instead.
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

        self._density_matrix = self._get_empty()

    def _get_empty(self) -> np.ndarray:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)
        return np.zeros(
            shape=(state_vector_size,) * 2, dtype=self._config.complex_dtype
        )

    def reset(self) -> None:
        self._density_matrix = self._get_empty()
        self._density_matrix[0, 0] = 1.0

    def _as_mixed(self):
        return self.copy()

    @classmethod
    def from_fock_state(cls, state: BaseFockState) -> "FockState":
        """Instantiation using another :class:`BaseFockState` instance.

        Args:
            state (BaseFockState):
                The instance from which a :class:`FockState` instance is created.
        """

        new_instance = cls(d=state.d, connector=state._connector, config=state._config)

        new_instance._density_matrix = state.density_matrix

        return new_instance

    @property
    def nonzero_elements(
        self,
    ) -> Generator[Tuple[complex, Tuple], Any, None]:
        np = self._connector.np

        nonzero_indices = np.nonzero(self._density_matrix)

        row_indices, col_indices = nonzero_indices

        row_occupation_numbers = self._space[row_indices]
        col_occupation_numbers = self._space[col_indices]

        nonzero_elements = self._density_matrix[nonzero_indices]

        for index, coefficient in enumerate(nonzero_elements):
            yield coefficient, (
                tuple(row_occupation_numbers[index]),
                tuple(col_occupation_numbers[index]),
            )

    def __str__(self) -> str:
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FockState):
            return False
        return np.allclose(self._density_matrix, other._density_matrix)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff)

        return self._density_matrix[:cardinality, :cardinality]

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        index = get_index_in_fock_space(occupation_number)
        return np.diag(self._density_matrix)[index].real

    @property
    def fock_probabilities(self) -> np.ndarray:
        return self._connector.np.real(self._connector.np.diag(self._density_matrix))

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        for index, basis in enumerate(self._space):
            probability_map[tuple(basis)] = np.abs(self._density_matrix[index, index])

        return probability_map

    def reduced(self, modes: Tuple[int, ...]) -> "FockState":
        np = self._connector.np
        fallback_np = self._connector.fallback_np
        d = self.d

        if modes == tuple(range(d)):
            return self

        outer_modes = self._get_auxiliary_modes(modes)

        inner_size = len(modes)
        outer_size = d - inner_size

        reduced_state = FockState(
            d=inner_size, connector=self._connector, config=self._config
        )

        density_list = [
            [0.0 for _ in range(reduced_state._density_matrix.shape[1])]
            for _ in range(reduced_state._density_matrix.shape[0])
        ]

        cutoff = self._config.cutoff
        inner_space = get_fock_space_basis(d=inner_size, cutoff=cutoff)
        outer_space = get_fock_space_basis(d=outer_size, cutoff=cutoff)

        index_list = []

        basis_matrix = fallback_np.empty(
            shape=(outer_space.shape[0], self.d), dtype=int
        )

        for inner_basis in inner_space:
            size = cutoff_fock_space_dim(
                cutoff=cutoff - fallback_np.sum(inner_basis), d=outer_size
            )

            basis_matrix[:size, modes] = inner_basis

            basis_matrix[:size, outer_modes] = outer_space[:size]

            partial_index_list = get_index_in_fock_space_array(basis_matrix[:size])

            index_list.append(partial_index_list)

        for i, ix1 in enumerate(index_list):
            for j, ix2 in enumerate(index_list):
                minlen = min(len(ix1), len(ix2))
                density_list[i][j] = np.sum(
                    self._density_matrix[ix1[:minlen], ix2[:minlen]]
                )

        reduced_state._density_matrix = np.array(density_list)

        return reduced_state

    @property
    def norm(self) -> float:
        np = self._connector.np
        return np.real(np.trace(self._density_matrix))

    def normalize(self) -> None:
        """Normalizes the density matrix to have a trace of 1.

        Raises:
            RuntimeError: Raised if the current norm of the state is too close to 0.
        """
        norm = self.norm

        if self._config.validate and np.isclose(norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._density_matrix = self._density_matrix / self.norm

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the density matrix is not positive semidefinite, not
                self-adjoint or the trace of the density matrix is not 1.
        """
        if not self._config.validate:
            return

        if not is_selfadjoint(self._density_matrix):
            raise InvalidState(
                "The density matrix is not self-adjoint:\n"
                f"density_matrix={self._density_matrix}"
            )

        fock_probabilities = self.fock_probabilities

        if not np.all(fock_probabilities >= 0.0):
            raise InvalidState(
                "The density matrix is not positive semidefinite.\n"
                f"fock_probabilities={fock_probabilities}"
            )

        trace_of_density_matrix = sum(fock_probabilities)

        if not np.isclose(trace_of_density_matrix, 1.0):
            raise InvalidState(
                f"The trace of the density matrix is {trace_of_density_matrix}, "
                "instead of 1.0:\n"
                f"fock_probabilities={fock_probabilities}"
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

        expectation = np.trace(np.dot(reduced_dm, rotated_quadratures)).real
        variance = (
            np.trace(
                np.dot(reduced_dm, np.dot(rotated_quadratures, rotated_quadratures))
            ).real
            - expectation**2
        )
        return expectation, variance

    def get_purity(self):
        np = self._connector.np
        density_matrix = self.density_matrix
        return np.real(np.einsum("ij,ji", density_matrix, density_matrix))
