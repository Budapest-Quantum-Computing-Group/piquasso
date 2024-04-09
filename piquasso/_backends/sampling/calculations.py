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

from functools import partial

import numpy as np
from piquasso._backends.sampling.state import SamplingState

from piquasso._math.validations import all_natural

from piquasso.instructions import gates

from piquasso.api.exceptions import InvalidState
from piquasso.api.result import Result
from piquasso.api.instruction import Instruction


from .utils import (
    generate_lossless_samples,
    generate_uniform_lossy_samples,
    generate_lossy_samples,
)


def state_vector(state: SamplingState, instruction: Instruction, shots: int) -> Result:
    if not np.all(state._initial_state == 0):
        raise InvalidState("State vector is already set.")

    coefficient = instruction._all_params["coefficient"]
    occupation_numbers = instruction._all_params["occupation_numbers"]

    initial_state = coefficient * np.array(occupation_numbers)

    if not all_natural(initial_state):
        raise InvalidState(
            f"Invalid initial state specified: instruction={instruction}"
        )

    state._initial_state = np.rint(initial_state).astype(int)

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
        matrix=instruction._get_passive_block(state._calculator, state._config),
        modes=instruction.modes,
    )

    return Result(state=state)


def _apply_matrix_on_modes(
    state: SamplingState, matrix: np.ndarray, modes: Tuple[int, ...]
) -> None:
    calculator = state._calculator
    np = calculator.np
    fallback_np = calculator.fallback_np

    embedded = np.identity(len(state.interferometer), dtype=state._config.complex_dtype)

    embedded = calculator.assign(embedded, fallback_np.ix_(modes, modes), matrix)

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
    `this article <https://arxiv.org/pdf/1612.01199.pdf>`_ for more details.

    This is a contribution implementation from `theboss`, see
    https://github.com/Tomev-CTP/theboss.

    This method assumes that initial_state is given in the second quantization
    description (mode occupation).

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.
    """

    initial_state = state._initial_state

    interferometer_svd = np.linalg.svd(state.interferometer)

    singular_values = interferometer_svd[1]

    if not state.is_lossy:
        calculate_permanent = partial(
            state._calculator.permanent, matrix=state.interferometer
        )
        samples = generate_lossless_samples(
            initial_state, shots, calculate_permanent, state._config.rng
        )
    elif np.all(np.isclose(singular_values, singular_values[0])):
        uniform_transmission_probability = singular_values[0] ** 2

        calculate_permanent = partial(
            state._calculator.permanent, matrix=state.interferometer
        )

        samples = generate_uniform_lossy_samples(
            initial_state,
            shots,
            calculate_permanent,
            uniform_transmission_probability,
            state._config.rng,
        )

    else:
        samples = generate_lossy_samples(
            initial_state,
            shots,
            state._calculator.permanent,
            interferometer_svd,
            state._config.rng,
        )

    return Result(state=state, samples=list(map(tuple, samples)))
