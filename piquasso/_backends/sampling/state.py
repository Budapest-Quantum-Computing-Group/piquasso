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

from typing import Tuple, List
import numpy as np

from piquasso._math.combinatorics import partitions
from piquasso._math.fock import symmetric_subspace_cardinality
from piquasso._math.linalg import is_unitary
from piquasso._math.validations import all_natural

from piquasso.api.config import Config
from piquasso.api.errors import InvalidState
from piquasso.api.state import State
from piquasso.api.errors import PiquassoException

from theboss.distribution_calculators.bs_distribution_calculator_with_fixed_losses import (  # noqa: E501
    BSDistributionCalculatorWithFixedLosses,
    BosonSamplingExperimentConfiguration,
)

from theboss.boson_sampling_utilities.permanent_calculators.ryser_guan_permanent_calculator import (  # noqa: E501
    RyserGuanPermanentCalculator,
)


class SamplingState(State):
    def __init__(self, d: int, config: Config = None) -> None:
        super().__init__(config=config)

        self.initial_state: np.ndarray = np.zeros((d,), dtype=int)
        self.interferometer: np.ndarray = np.diag(np.ones(d, dtype=complex))

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
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        if len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        number_of_particles = sum(occupation_number)

        if number_of_particles != self.particle_number:
            return 0.0

        basis = partitions(self.d, number_of_particles)

        index = basis.index(occupation_number)

        subspace_probabilities = self._get_fock_probabilities_on_subspace()

        return subspace_probabilities[index]

    @property
    def fock_probabilities(self) -> np.ndarray:
        # TODO: All the `fock_probabilities` properties return a list according to the
        # `cutoff` specified in `config`. However, here it does not make sense to adhere
        # to that...

        cutoff = self.particle_number + 1

        probabilities = []

        for particle_number in range(cutoff):

            if particle_number == self.particle_number:
                subspace_probabilities = self._get_fock_probabilities_on_subspace()
            else:
                cardinality = symmetric_subspace_cardinality(
                    d=self.d, n=particle_number
                )
                subspace_probabilities = [0.0] * cardinality

            probabilities.extend(subspace_probabilities)

        return np.array(probabilities)

    def _get_fock_probabilities_on_subspace(self) -> List[float]:
        """
        The order of the returned Fock states is lexicographic, according to
        `theboss.boson_sampling_utilities.boson_sampling_utilities
        .generate_possible_outputs`
        """
        permanent_calculator = RyserGuanPermanentCalculator(self.interferometer)
        config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.interferometer,
            initial_state=np.asarray(self.initial_state),
            number_of_modes=self.d,
            initial_number_of_particles=self.particle_number,
            number_of_particles_lost=0,
            number_of_particles_left=self.particle_number,
        )
        distribution_calculator = BSDistributionCalculatorWithFixedLosses(
            config, permanent_calculator
        )
        return distribution_calculator.calculate_distribution()
