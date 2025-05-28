#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

from typing import Optional, TYPE_CHECKING

from .._utils import (
    get_cutoff_fock_space_dimension,
    get_fock_space_index,
    get_fock_space_basis,
)

from piquasso.api.exceptions import InvalidState

from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.connector import BaseConnector

if TYPE_CHECKING:
    import numpy as np


class PureFockState(State):
    r"""A fermionic pure Fock state."""

    def __init__(
        self,
        d: int,
        connector: BaseConnector,
        config: Optional[Config] = None,
    ) -> None:
        self._d = d

        super().__init__(connector=connector, config=config)

        dim = get_cutoff_fock_space_dimension(d, self._config.cutoff)

        self._state_vector = connector.np.zeros(
            shape=dim, dtype=self._config.complex_dtype
        )

    @property
    def state_vector(self):
        """The state vector of the quantum state.

        .. warning::
            The primary ordering of the Fock basis is by number of particles, and the
            secondary is anti-lexicographic.

            Example for 3 modes:

            .. math ::

                \ket{000},
                \ket{100}, \ket{010}, \ket{001},
                \ket{200}, \ket{110}, \ket{101}, \ket{020}, \ket{011}, \ket{002}, \dots

            This is in contrast with
            :meth:`~piquasso.fermionic.gaussian.state.GaussianState.density_matrix`,
            where the primary ordering is lexicographic.
        """
        return self._state_vector

    @property
    def d(self):
        return self._d

    @property
    def fock_probabilities(self) -> "np.ndarray":
        np = self._connector.np
        return np.conj(self._state_vector) * self._state_vector

    @property
    def fock_probabilities_map(self) -> dict:
        fock_space_basis = get_fock_space_basis(self.d, self._config.cutoff)

        occupation_numbers = [tuple(x) for x in fock_space_basis.tolist()]

        return dict(zip(occupation_numbers, self.fock_probabilities))

    @property
    def norm(self) -> float:
        return self._connector.np.sum(self.fock_probabilities)

    def get_particle_detection_probability(self, occupation_number):
        index = get_fock_space_index(
            self._connector.fallback_np.array(occupation_number, dtype=int)
        )

        return self._connector.np.abs(self._state_vector[index]) ** 2

    def validate(self):
        if not self._config.validate:
            return

        np = self._connector.np

        norm = self.norm

        if not np.isclose(norm, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {norm}, instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )

    @property
    def density_matrix(self):
        return self._connector.np.outer(self._state_vector, self._state_vector.conj())

    def __eq__(self, other):
        if not isinstance(other, PureFockState):
            return False

        return self._np.allclose(self.state_vector, other.state_vector)
