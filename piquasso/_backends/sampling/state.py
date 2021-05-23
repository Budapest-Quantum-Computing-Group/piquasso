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

import numpy as np

from piquasso.api.state import State
from piquasso.api.errors import InvalidState
from piquasso._math.fock import symmetric_subspace_cardinality
from piquasso._math.linalg import is_unitary

from BoSS.distribution_calculators.bs_distribution_calculator_with_fixed_losses import (
    BSDistributionCalculatorWithFixedLosses,
    BosonSamplingExperimentConfiguration
)
from BoSS.boson_sampling_utilities.permanent_calculators.\
    ryser_guan_permanent_calculator import RyserGuanPermanentCalculator
from .circuit import SamplingCircuit


class SamplingState(State):
    circuit_class = SamplingCircuit

    def __init__(self, *initial_state):
        self.initial_state = initial_state
        self.interferometer = np.diag(np.ones(self.d, dtype=complex))
        self.results = None

        self.is_lossy = False

    def validate(self):
        """Validates the currect state.

        Raises:
            InvalidState: If the interferometer matrix is non-unitary.
        """

        if not is_unitary(self.interferometer):
            raise InvalidState("The interferometer matrix is not unitary.")

    def _apply_passive_linear(self, U, modes):
        r"""
        Multiplies the interferometer of the state with the `U` matrix (representing
        the additional interferometer) in the qumodes specified in `modes`.

        The size of `U` should be smaller than or equal to the size of the
        interferometer.

        The `modes` can contain any number in any order as long as number of qumodes is
        equal to the size of the `U` matrix

        Args:
            U (numpy.ndarray): A square matrix to multiply to the interferometer.
            modes (tuple[int]):
                Distinct positive integer values which are used to represent qumodes.
        """
        self._apply_matrix_on_modes(U, modes)

    def _apply_loss(self, transmissivity, modes):
        self.is_lossy = True

        transmission_matrix = np.diag(transmissivity)

        self._apply_matrix_on_modes(transmission_matrix, modes)

    def _apply_matrix_on_modes(self, matrix, modes):
        embedded = np.identity(len(self.interferometer), dtype=complex)

        embedded[np.ix_(modes, modes)] = matrix

        self.interferometer = embedded @ self.interferometer

    @property
    def d(self) -> int:
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.initial_state)

    @property
    def particle_number(self) -> int:
        r"""The number of particles in the system.

        Returns:
            int: The number of particles.
        """

        return sum(self.initial_state)

    def get_fock_probabilities(self, cutoff: int = None) -> list:
        cutoff = cutoff or self.particle_number + 1

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

        return probabilities

    def _get_fock_probabilities_on_subspace(self) -> list:
        """
        The order if the returned Fock states is lexicographic, according to
        `BoSS.boson_sampling_utilities.boson_sampling_utilities
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
