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
    """A state dedicated for Boson Sampling (or related) simulations.

    When using :class:`~piquasso._simulators.sampling.simulator.SamplingSimulator`, the
    simulation results will contain an instance of this class, containing the input
    occupation numbers and the interferometer to be applied.

    Example usage:

    .. code-block:: python

        >>> import piquasso as pq
        >>>
        >>> from scipy.stats import unitary_group
        >>>
        >>> d = 7
        >>>
        >>> interferometer_matrix = unitary_group.rvs(d)
        >>>
        >>> with pq.Program() as program:
        >>>     pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0, 0, 0])
        >>>     pq.Q(all) | pq.Interferometer(interferometer_matrix)
        >>>     pq.Q(all) | pq.ParticleNumberMeasurement()
        >>>
        >>> simulator = pq.SamplingSimulator(d=d)
        >>>
        >>> result = simulator.execute(program, shots=3)
        >>>
        >>> result.samples
        [(0, 0, 0, 0, 2, 1, 0), (3, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 2, 1)]
    """

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
        """The interferometer matrix corresponding to the circuit."""

        self.is_lossy = False
        """Returns `True` if the state is lossy, otherwise `False`."""

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
        """The norm of the state."""
        return np.sum(np.abs(self._coefficients) ** 2)

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        np = self._connector.np
        fallback_np = self._connector.fallback_np

        output_number_of_particles = fallback_np.sum(occupation_number)

        sum_ = 0.0

        for index, input_occupation_number in enumerate(self._occupation_numbers):
            if output_number_of_particles != fallback_np.sum(input_occupation_number):
                continue

            inner_product = calculate_inner_product(
                interferometer=self.interferometer,
                input=input_occupation_number,
                output=fallback_np.array(occupation_number),
                connector=self._connector,
            )
            coefficient = self._coefficients[index]

            sum_ += coefficient * inner_product

        return np.abs(sum_) ** 2

    @property
    def state_vector(self):
        """The state vector representation of this state.

        This implementation follows Algorithm 1 (`SLOS_full`) from
        `Strong Simulation of Linear Optical Processes <https://arxiv.org/pdf/2206.10549>`_.
        """  # noqa: E501

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

            index_range = fallback_np.arange(starting_index, ending_index)

            state_vector = connector.assign(
                state_vector,
                index_range,
                state_vector[index_range] + coefficient * partial_state_vector,
            )

        return state_vector

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix of the state in the truncated Fock space."""
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
