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
from piquasso.api.result import Result
from piquasso.api.instruction import Instruction

from BoSS.distribution_calculators.bs_distribution_calculator_with_fixed_losses import (
    BSDistributionCalculatorWithFixedLosses,
    BosonSamplingExperimentConfiguration,
)
from BoSS.boson_sampling_simulator import BosonSamplingSimulator

# The fastest implemented permanent calculator is currently Ryser-Guan
from BoSS.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import (  # noqa: E501
    BSPermanentCalculatorInterface,
)
from BoSS.boson_sampling_utilities.permanent_calculators.ryser_guan_permanent_calculator import (  # noqa: E501
    RyserGuanPermanentCalculator,
)

# Fastest boson sampling algorithm generalized for bunched states
from BoSS.simulation_strategies.generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)
from BoSS.simulation_strategies.generalized_cliffords_uniform_losses_simulation_strategy import (  # noqa: E501
    GeneralizedCliffordsUniformLossesSimulationStrategy,
)

# Fastest BS algorithm generalized for bunched states, but with lossy network
from BoSS.simulation_strategies.lossy_networks_generalized_cliffords_simulation_strategy import (  # noqa: E501
    LossyNetworksGeneralizedCliffordsSimulationStrategy,
)
from BoSS.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
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

    def _state_vector(self, instruction: Instruction) -> None:
        if not np.all(self.initial_state == 0):
            raise InvalidState("State vector is already set.")

        coefficient = instruction._all_params["coefficient"]
        occupation_numbers = instruction._all_params["occupation_numbers"]

        initial_state = coefficient * np.array(occupation_numbers)

        if not all_natural(initial_state):
            raise InvalidState(
                f"Invalid initial state specified: instruction={instruction}"
            )

        self.initial_state = np.rint(initial_state).astype(int)

    def _passive_linear(self, instruction: Instruction) -> None:
        r"""Applies an interferometer to the circuit.

        This can be interpreted as placing another interferometer in the network, just
        before performing the sampling. This instruction is realized by multiplying
        current effective interferometer matrix with new interferometer matrix.

        Do note, that new interferometer matrix works as interferometer matrix on
        qumodes (provided as the arguments) and as an identity on every other mode.
        """
        self._apply_matrix_on_modes(
            matrix=instruction._all_params["passive_block"], modes=instruction.modes
        )

    def _loss(self, instruction: Instruction) -> None:
        self.is_lossy = True

        self._apply_matrix_on_modes(
            matrix=np.diag(instruction._all_params["transmissivity"]),
            modes=instruction.modes,
        )

    def _apply_matrix_on_modes(
        self, matrix: np.ndarray, modes: Tuple[int, ...]
    ) -> None:
        embedded = np.identity(len(self.interferometer), dtype=complex)

        embedded[np.ix_(modes, modes)] = matrix

        self.interferometer = embedded @ self.interferometer

    def _get_sampling_simulation_strategy(
        self, permanent_calculator: BSPermanentCalculatorInterface
    ) -> SimulationStrategyInterface:
        if not self.is_lossy:
            return GeneralizedCliffordsSimulationStrategy(permanent_calculator)

        _, singular_values, _ = np.linalg.svd(self.interferometer)

        if np.all(np.isclose(singular_values, singular_values[0])):
            return GeneralizedCliffordsUniformLossesSimulationStrategy(
                permanent_calculator
            )

        return LossyNetworksGeneralizedCliffordsSimulationStrategy(permanent_calculator)

    def _sampling(self, instruction: Instruction) -> None:
        initial_state = np.array(self.initial_state)
        permanent_calculator = RyserGuanPermanentCalculator(
            matrix=self.interferometer, input_state=initial_state
        )

        simulation_strategy = self._get_sampling_simulation_strategy(
            permanent_calculator
        )

        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        samples = sampling_simulator.get_classical_simulation_results(
            initial_state, samples_number=self.shots
        )

        self.result = Result(samples=list(map(tuple, samples)))

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
        number_of_particles = sum(occupation_number)

        if number_of_particles != self.particle_number:
            return 0.0

        basis = partitions(self.d, number_of_particles)

        index = basis.index(occupation_number)

        subspace_probabilities = self._get_fock_probabilities_on_subspace()

        return subspace_probabilities[index]

    @property
    def fock_probabilities(self) -> List[float]:
        """
        TODO: All the `fock_probabilities` properties return a list according to the
        `cutoff` specified in `config`. However, here it does not make sense to adhere
        to that...
        """

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

        return probabilities

    def _get_fock_probabilities_on_subspace(self) -> List[float]:
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
