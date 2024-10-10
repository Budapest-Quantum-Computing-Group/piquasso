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

from typing import Optional, List
import numpy as np

from piquasso._math.fock import cutoff_fock_space_dim
from piquasso._math.linalg import is_unitary

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState
from piquasso.api.state import State
from piquasso.api.exceptions import PiquassoException
from piquasso.api.connector import BaseConnector

from .utils import (
    calculate_state_vector,
    calculate_inner_product,
)


class SamplingState(State):
    def __init__(
        self, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            connector (BaseConnector): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """
        super().__init__(connector=connector, config=config)

        self._d = d

        self._occupation_numbers: List = []
        self._coefficients: List = []
        self.interferometer = np.diag(np.ones(d, dtype=self._config.complex_dtype))

        self.is_lossy = False

    def validate(self) -> None:
        """Validates the current state.

        Raises:
            InvalidState: If the interferometer matrix is non-unitary, or the input
                state is invalid.
        """
        if not self._config.validate:
            return

        if not is_unitary(self.interferometer):
            raise InvalidState("The interferometer matrix is not unitary.")

        for occupation_number in self._occupation_numbers:
            if len(occupation_number) != self.d:
                raise InvalidState(
                    f"The occupation number '{occupation_number}' is not well-defined "
                    f"on '{self.d}' modes."
                )

        norm = self.norm

        if not np.isclose(norm, 1.0):
            raise InvalidState(f"The state is not normalized: norm={norm}")

    @property
    def d(self):
        return self._d

    @property
    def norm(self):
        return np.sum(np.abs(self._coefficients) ** 2)

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        output_number_of_particles = np.sum(occupation_number)

        sum_ = 0.0

        for index, input_occupation_number in enumerate(self._occupation_numbers):
            if output_number_of_particles != np.sum(input_occupation_number):
                continue

            inner_product = calculate_inner_product(
                interferometer=self.interferometer,
                input=input_occupation_number,
                output=np.array(occupation_number),
                connector=self._connector,
            )
            coefficient = self._coefficients[index]

            sum_ += coefficient * inner_product

        return np.abs(sum_) ** 2

    @property
    def state_vector(self):
        connector = self._connector
        np = connector.np
        fallback_np = connector.fallback_np

        particle_numbers = fallback_np.sum(self._occupation_numbers, axis=1)

        state_vector = np.zeros(
            shape=cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff),
            dtype=self._config.complex_dtype,
        )

        for index in range(len(self._occupation_numbers)):
            particle_number = particle_numbers[index]

            starting_index = cutoff_fock_space_dim(d=self.d, cutoff=particle_number)

            ending_index = cutoff_fock_space_dim(d=self.d, cutoff=particle_number + 1)

            coefficient = self._coefficients[index]

            input_state = self._occupation_numbers[index]

            partial_state_vector = calculate_state_vector(
                self.interferometer, input_state, self._config, self._connector
            )

            index = fallback_np.arange(starting_index, ending_index)

            state_vector = connector.assign(
                state_vector,
                index,
                state_vector[index] + coefficient * partial_state_vector,
            )

        return state_vector

    @property
    def density_matrix(self) -> np.ndarray:
        state_vector = self.state_vector

        return self._np.outer(state_vector, self._np.conj(state_vector))

    @property
    def fock_probabilities(self) -> np.ndarray:
        np = self._connector.np

        state_vector = self.state_vector

        return np.real(np.conj(state_vector) * state_vector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplingState):
            return False

        return (
            np.allclose(self._coefficients, other._coefficients)
            and np.allclose(self._occupation_numbers, other._occupation_numbers)
            and np.allclose(self.interferometer, other.interferometer)
            and self.is_lossy == other.is_lossy
        )
