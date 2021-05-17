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
        self.interferometer[np.ix_(modes, modes)] = (
            matrix @ self.interferometer[np.ix_(modes, modes)]
        )

    @property
    def d(self):
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.initial_state)

    def get_fock_probabilities(self):
        permanent_calculator = RyserGuanPermanentCalculator(self.interferometer)
        config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.interferometer,
            initial_state=np.asarray(self.initial_state),
            number_of_modes=len(self.initial_state),
            initial_number_of_particles=sum(self.initial_state),
            number_of_particles_lost=0,
            number_of_particles_left=sum(self.initial_state)
        )
        distribution_calculator = BSDistributionCalculatorWithFixedLosses(
            config, permanent_calculator)
        # The order of the probabilities is according to
        # BoSS.boson_sampling_utilities.boson_sampling_utilities
        # .generate_possible_outputs
        return distribution_calculator.calculate_distribution()
