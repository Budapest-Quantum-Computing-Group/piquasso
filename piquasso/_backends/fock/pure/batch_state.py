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

from typing import Optional

import numpy as np

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState
from piquasso.api.calculator import BaseCalculator

from piquasso._math.linalg import vector_absolute_square
from piquasso._math.indices import get_index_in_fock_space

from .state import PureFockState


class BatchPureFockState(PureFockState):
    r"""A simulated batch pure Fock state, containing multiple state vectors."""

    def __init__(
        self, *, d: int, calculator: BaseCalculator, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            calculator (BaseCalculator): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """

        super().__init__(d=d, calculator=calculator, config=config)

    def _apply_separate_state_vectors(self, state_vectors):
        self._state_vector = self._np.array(
            state_vectors, dtype=self._config.complex_dtype
        ).T

    @property
    def _batch_size(self):
        return self._state_vector.shape[1]

    @property
    def _batch_state_vectors(self):
        for index in range(self._batch_size):
            yield self._state_vector[:, index]

    @property
    def nonzero_elements(self):
        return [
            self._nonzero_elements_for_single_state_vector(state_vector)
            for state_vector in self._batch_state_vectors
        ]

    def __repr__(self) -> str:
        partial_strings = []
        for partial_nonzero_elements in self.nonzero_elements:
            partial_strings.append(
                self._get_repr_for_single_state_vector(partial_nonzero_elements)
            )

        return "\n".join(partial_strings)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BatchPureFockState):
            return False
        return self._np.allclose(self._state_vector, other._state_vector)

    @property
    def fock_probabilities(self):
        return [
            vector_absolute_square(state_vector, self._calculator)
            for state_vector in self._batch_state_vectors
        ]

    @property
    def norm(self):
        return [
            self._calculator.np.sum(partial_fock_probabilities)
            for partial_fock_probabilities in self.fock_probabilities
        ]

    def normalize(self) -> None:
        if not self._config.normalize:
            return

        norms = self.norm

        if any(np.isclose(norm, 0) for norm in norms):
            raise InvalidState("The norm of a state in the batch is 0.")

        self._state_vector = self._state_vector / self._np.sqrt(norms)

    def validate(self) -> None:
        if not all(np.isclose(norm, 1.0) for norm in self.norm):
            raise InvalidState(
                "The sum of probabilities is not close to 1.0 for at least one state "
                "in the batch."
            )

    def _get_mean_position_indices(self, mode):
        fallback_np = self._calculator.fallback_np

        left_indices = []
        multipliers = []
        right_indices = []

        for index, basis in enumerate(self._space):
            i = basis[mode]
            basis_array = fallback_np.array(basis)

            if i > 0:
                basis_array[mode] = i - 1
                lower_index = get_index_in_fock_space(tuple(basis_array))

                left_indices.append(lower_index)
                multipliers.append(fallback_np.sqrt(i))
                right_indices.append(index)

            if sum(basis) + 1 < self._config.cutoff:
                basis_array[mode] = i + 1
                upper_index = get_index_in_fock_space(tuple(basis_array))

                left_indices.append(upper_index)
                multipliers.append(fallback_np.sqrt(i + 1))
                right_indices.append(index)

        multipliers = fallback_np.array(multipliers)

        return multipliers, left_indices, right_indices

    def mean_position(self, mode: int) -> np.ndarray:
        np = self._calculator.np
        fallback_np = self._calculator.fallback_np
        multipliers, left_indices, right_indices = self._get_mean_position_indices(mode)

        lhs = (multipliers * self._state_vector[left_indices].T).T
        rhs = self._state_vector[right_indices]

        return np.real(
            np.einsum("ij,ij->j", lhs, rhs) * fallback_np.sqrt(self._config.hbar / 2)
        )
