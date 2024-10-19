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

from typing import Tuple

import numpy as np
from piquasso._simulators.sampling.state import SamplingState

from piquasso.instructions import gates

from piquasso.api.exceptions import InvalidState, NotImplementedCalculation
from piquasso.api.result import Result
from piquasso.api.instruction import Instruction

from piquasso._simulators.fock.pure.calculations import (
    post_select_photons as pure_fock_post_select_photons,
    imperfect_post_select_photons as pure_fock_imperfect_post_select_photons,
)


from .utils import (
    generate_lossless_samples,
    generate_uniform_lossy_samples,
    generate_lossy_samples,
)


def state_vector(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    coefficient = instruction._all_params["coefficient"]
    occupation_numbers = instruction._all_params["occupation_numbers"]

    if state._config.validate and len(occupation_numbers) != state.d:
        raise InvalidState(
            f"The occupation numbers '{occupation_numbers}' are not well-defined "
            f"on '{state.d}' modes: instruction={instruction}"
        )

    state._occupation_numbers.append(np.rint(occupation_numbers).astype(int))
    state._coefficients.append(coefficient)

    return Result(state=state)


def passive_linear(
    state: SamplingState, instruction: gates._PassiveLinearGate, shots: int
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
        matrix=instruction._get_passive_block(state._connector, state._config),
        modes=instruction.modes,
    )

    return Result(state=state)


def _apply_matrix_on_modes(
    state: SamplingState, matrix: np.ndarray, modes: Tuple[int, ...]
) -> None:
    connector = state._connector
    np = connector.np
    fallback_np = connector.fallback_np

    embedded = np.identity(len(state.interferometer), dtype=state._config.complex_dtype)

    embedded = connector.assign(embedded, fallback_np.ix_(modes, modes), matrix)

    state.interferometer = embedded @ state.interferometer


def loss(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    state.is_lossy = True

    _apply_matrix_on_modes(
        state=state,
        matrix=np.array([[instruction._all_params["transmissivity"]]]),
        modes=instruction.modes,
    )

    return Result(state=state)


def lossy_interferometer(
    state: SamplingState, instruction: Instruction, shots: int
) -> Result:
    state.is_lossy = True

    _apply_matrix_on_modes(
        state=state,
        matrix=instruction._all_params["matrix"],
        modes=instruction.modes,
    )

    return Result(state=state)


def particle_number_measurement(
    state: SamplingState, instruction: Instruction, shots: int
) -> Result:
    """
    Simulates a boson sampling using generalized Clifford & Clifford algorithm
    from [Brod, Oszmaniec 2020] see
    `this article <https://arxiv.org/abs/1906.06696>`_ for more details.

    This is a contribution implementation from `theboss`, see
    https://github.com/Tomev-CTP/theboss.

    This method assumes that initial_state is given in the second quantization
    description (mode occupation).

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.
    """

    if (
        state._config.validate
        and len(state._occupation_numbers) != 1
        and not np.isclose(state._coefficients[0], 1.0)
    ):
        raise NotImplementedCalculation(
            f"The instruction {instruction} is not supported for states defined using "
            "multiple 'StateVector' instructions.\n"
            "If you need this feature to be implemented, please create an issue at "
            "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
        )

    initial_state = state._occupation_numbers[0]

    interferometer_svd = np.linalg.svd(state.interferometer)

    singular_values = interferometer_svd[1]

    if not state.is_lossy:
        samples = generate_lossless_samples(
            initial_state,
            shots,
            state._connector.permanent,
            state.interferometer,
            state._config.rng,
        )
    elif np.all(np.isclose(singular_values, singular_values[0])):
        uniform_transmission_probability = singular_values[0] ** 2

        samples = generate_uniform_lossy_samples(
            initial_state,
            shots,
            state._connector.permanent,
            state.interferometer,
            uniform_transmission_probability,
            state._config.rng,
        )

    else:
        samples = generate_lossy_samples(
            initial_state,
            shots,
            state._connector.permanent,
            interferometer_svd,
            state._config.rng,
        )

    return Result(state=state, samples=list(map(tuple, samples)))


post_select_photons = pure_fock_post_select_photons
imperfect_post_select_photons = pure_fock_imperfect_post_select_photons
