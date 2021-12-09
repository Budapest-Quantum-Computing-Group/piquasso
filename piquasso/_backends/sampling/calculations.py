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

from typing import Tuple

import numpy as np
from piquasso._backends.sampling.state import SamplingState

from piquasso._math.validations import all_natural

from piquasso.api.errors import InvalidState
from piquasso.api.result import Result
from piquasso.api.instruction import Instruction

from theboss.boson_sampling_simulator import BosonSamplingSimulator

# The fastest implemented permanent calculator is currently Ryser-Guan
from theboss.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import (  # noqa: E501
    BSPermanentCalculatorInterface,
)
from theboss.boson_sampling_utilities.permanent_calculators.ryser_guan_permanent_calculator import (  # noqa: E501
    RyserGuanPermanentCalculator,
)

# Fastest boson sampling algorithm generalized for bunched states
from theboss.simulation_strategies.generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)
from theboss.simulation_strategies.generalized_cliffords_uniform_losses_simulation_strategy import (  # noqa: E501
    GeneralizedCliffordsUniformLossesSimulationStrategy,
)

# Fastest BS algorithm generalized for bunched states, but with lossy network
from theboss.simulation_strategies.lossy_networks_generalized_cliffords_simulation_strategy import (  # noqa: E501
    LossyNetworksGeneralizedCliffordsSimulationStrategy,
)
from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


def state_vector(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    if not np.all(state.initial_state == 0):
        raise InvalidState("State vector is already set.")

    coefficient = instruction._all_params["coefficient"]
    occupation_numbers = instruction._all_params["occupation_numbers"]

    initial_state = coefficient * np.array(occupation_numbers)

    if not all_natural(initial_state):
        raise InvalidState(
            f"Invalid initial state specified: instruction={instruction}"
        )

    state.initial_state = np.rint(initial_state).astype(int)

    return Result(state=state)


def passive_linear(
    state: SamplingState, instruction: Instruction, shots: int
) -> Result:
    r"""Applies an interferometer to the circuit.

    This can be interpreted as placing another interferometer in the network, just
    before performing the sampling. This instruction is realized by multiplying
    current effective interferometer matrix with new interferometer matrix.

    Do note, that new interferometer matrix works as interferometer matrix on
    qumodes (provided as the arguments) and as an identity on every other mode.
    """
    _apply_matrix_on_modes(
        state=state,
        matrix=instruction._all_params["passive_block"],
        modes=instruction.modes,
    )

    return Result(state=state)


def _apply_matrix_on_modes(
    state: SamplingState, matrix: np.ndarray, modes: Tuple[int, ...]
) -> None:
    embedded = np.identity(len(state.interferometer), dtype=complex)

    embedded[np.ix_(modes, modes)] = matrix

    state.interferometer = embedded @ state.interferometer


def loss(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    state.is_lossy = True

    _apply_matrix_on_modes(
        state=state,
        matrix=np.diag(instruction._all_params["transmissivity"]),
        modes=instruction.modes,
    )

    return Result(state=state)


def sampling(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    initial_state = np.array(state.initial_state)
    permanent_calculator = RyserGuanPermanentCalculator(
        matrix=state.interferometer, input_state=initial_state
    )

    simulation_strategy = _get_sampling_simulation_strategy(state, permanent_calculator)

    sampling_simulator = BosonSamplingSimulator(simulation_strategy)

    samples = sampling_simulator.get_classical_simulation_results(
        initial_state, samples_number=shots
    )

    return Result(state=state, samples=list(map(tuple, samples)))


def _get_sampling_simulation_strategy(
    state: SamplingState, permanent_calculator: BSPermanentCalculatorInterface
) -> SimulationStrategyInterface:
    if not state.is_lossy:
        return GeneralizedCliffordsSimulationStrategy(permanent_calculator)

    _, singular_values, _ = np.linalg.svd(state.interferometer)

    if np.all(np.isclose(singular_values, singular_values[0])):
        return GeneralizedCliffordsUniformLossesSimulationStrategy(permanent_calculator)

    return LossyNetworksGeneralizedCliffordsSimulationStrategy(permanent_calculator)
