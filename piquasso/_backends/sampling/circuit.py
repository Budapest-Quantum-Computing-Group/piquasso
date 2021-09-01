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
import typing

from BoSS.boson_sampling_simulator import BosonSamplingSimulator
# The fastest implemented permanent calculator is currently Ryser-Guan
from BoSS.boson_sampling_utilities.permanent_calculators\
    .bs_permanent_calculator_interface import \
    BSPermanentCalculatorInterface
from BoSS.boson_sampling_utilities.permanent_calculators. \
    ryser_guan_permanent_calculator import RyserGuanPermanentCalculator
# Fastest boson sampling algorithm generalized for bunched states
from BoSS.simulation_strategies.generalized_cliffords_simulation_strategy import \
    GeneralizedCliffordsSimulationStrategy
from BoSS.simulation_strategies. \
    generalized_cliffords_uniform_losses_simulation_strategy import \
    GeneralizedCliffordsUniformLossesSimulationStrategy
# Fastest BS algorithm generalized for bunched states, but with lossy network
from BoSS.simulation_strategies. \
    lossy_networks_generalized_cliffords_simulation_strategy import \
    LossyNetworksGeneralizedCliffordsSimulationStrategy
from BoSS.simulation_strategies.simulation_strategy_interface import \
    SimulationStrategyInterface

from piquasso.api.circuit import Circuit
from piquasso.api.result import Result
from piquasso.api.instruction import Instruction

if typing.TYPE_CHECKING:
    from . import SamplingState


class SamplingCircuit(Circuit):
    r"""Circuit for Boson Sampling."""

    instruction_map = {
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "Sampling": "_sampling",
        "Interferometer": "_passive_linear",
        "Loss": "_loss",
    }

    def _passive_linear(self, instruction: Instruction, state: "SamplingState") -> None:
        r"""Applies an interferometer to the circuit.

        This can be interpreted as placing another interferometer in the network, just
        before performing the sampling. This instruction is realized by multiplying
        current effective interferometer matrix with new interferometer matrix.

        Do note, that new interferometer matrix works as interferometer matrix on
        qumodes (provided as the arguments) and as an identity on every other mode.
        """

        state._apply_passive_linear(
            instruction._all_params["passive_block"],
            instruction.modes,
        )

    @staticmethod
    def _get_sampling_simulation_strategy(
        state: "SamplingState",
        permanent_calculator: BSPermanentCalculatorInterface
    ) -> SimulationStrategyInterface:
        if not state.is_lossy:
            return GeneralizedCliffordsSimulationStrategy(permanent_calculator)

        _, singular_values, _ = np.linalg.svd(state.interferometer)

        if np.all(np.isclose(singular_values, singular_values[0])):
            return GeneralizedCliffordsUniformLossesSimulationStrategy(
                permanent_calculator
            )

        return LossyNetworksGeneralizedCliffordsSimulationStrategy(permanent_calculator)

    def _sampling(self, instruction: Instruction, state: "SamplingState") -> None:
        initial_state = np.array(state.initial_state)
        permanent_calculator = RyserGuanPermanentCalculator(
            matrix=state.interferometer, input_state=initial_state)

        simulation_strategy = self._get_sampling_simulation_strategy(
            state, permanent_calculator
        )

        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        samples = sampling_simulator.get_classical_simulation_results(
            initial_state,
            samples_number=self.shots
        )

        self.result = Result(instruction=instruction, samples=samples)

    def _loss(self, instruction: Instruction, state: "SamplingState") -> None:
        state._apply_loss(
            transmissivity=instruction._all_params["transmissivity"],
            modes=instruction.modes,
        )
