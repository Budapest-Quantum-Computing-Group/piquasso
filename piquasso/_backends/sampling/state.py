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

from piquasso._math.indices import get_index_in_fock_subspace
from piquasso._math.fock import cutoff_cardinality
from piquasso._math.linalg import is_unitary
from piquasso._math.validations import all_natural

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState
from piquasso.api.state import State
from piquasso.api.exceptions import PiquassoException
from piquasso.api.calculator import BaseCalculator

from .utils import calculate_distribution, calculate_state_vector


class SamplingState(State):
    def __init__(
        self, d: int, calculator: BaseCalculator, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            calculator (BaseCalculator): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """
        super().__init__(calculator=calculator, config=config)

        self.initial_state = np.zeros((d,), dtype=int)
        self.interferometer = np.diag(np.ones(d, dtype=self._config.complex_dtype))

        self.is_lossy = False

    def validate(self) -> None:
        """Validates the currect state.

        Raises:
            InvalidState: If the interferometer matrix is non-unitary.
        """

        if not is_unitary(self.interferometer):
            raise InvalidState("The interferometer matrix is not unitary.")

        if not all_natural(self.initial_state):
            raise InvalidState(
                f"Invalid initial state: initial_state={self.initial_state}"
            )

    @property
    def d(self) -> int:
        r"""The number of modes, on which the state is defined."""
        return len(self.initial_state)

    @property
    def particle_number(self) -> int:
        r"""The number of particles in the system."""
        return sum(self.initial_state)

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        number_of_particles = np.sum(occupation_number)

        if number_of_particles != self.particle_number:
            return 0.0

        index = get_index_in_fock_subspace(occupation_number)

        subspace_probabilities = self._get_fock_probabilities_on_subspace()

        return subspace_probabilities[index]

    @property
    def state_vector(self):
        np = self._calculator.np
        state_vector_on_smaller_subspaces = np.zeros(
            shape=cutoff_cardinality(d=self.d, cutoff=self.particle_number),
            dtype=self._config.dtype,
        )

        partial_state_vector = calculate_state_vector(
            self.interferometer, self.initial_state, self._config, self._calculator
        )

        return np.concatenate([state_vector_on_smaller_subspaces, partial_state_vector])

    @property
    def fock_probabilities(self) -> np.ndarray:
        # TODO: All the `fock_probabilities` properties return a list according to the
        # `cutoff` specified in `config`. However, here it does not make sense to adhere
        # to that...

        probabilities_on_smaller_subspaces: np.ndarray = np.zeros(
            shape=cutoff_cardinality(d=self.d, cutoff=self.particle_number),
            dtype=self._config.dtype,
        )
        return np.concatenate(
            [
                probabilities_on_smaller_subspaces,
                self._get_fock_probabilities_on_subspace(),
            ]
        )

    def _get_fock_probabilities_on_subspace(self) -> List[float]:
        # NOTE: The order of the returned Fock states is anti-lexicographic.
        return calculate_distribution(
            self.interferometer, self.initial_state, self._config, self._calculator
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplingState):
            return False

        return (
            np.allclose(self.initial_state, other.initial_state)
            and np.allclose(self.interferometer, other.interferometer)
            and self.is_lossy == other.is_lossy
        )
