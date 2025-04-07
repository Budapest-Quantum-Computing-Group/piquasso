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

from piquasso.api.result import Result

from piquasso.api.exceptions import InvalidParameter

from piquasso._math.validations import all_zero_or_one, are_modes_consecutive

from ._utils import (
    calculate_indices_for_controlled_phase,
    calculate_indices_for_ising_XX,
)

from .._utils import (
    get_fock_space_index,
    get_cutoff_fock_space_dimension,
    next_second_quantized,
)
from .state import PureFockState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piquasso.instructions.preparations import StateVector
    from piquasso.instructions.gates import _PassiveLinearGate, Squeezing2
    from piquasso.fermionic.instructions import ControlledPhase, IsingXX


def state_vector(
    state: PureFockState, instruction: "StateVector", shots: int
) -> Result:
    connector = state._connector

    fallback_np = connector.fallback_np

    coefficient = instruction._all_params["coefficient"]
    occupation_numbers = fallback_np.array(
        instruction._all_params["occupation_numbers"]
    )

    if state._config.validate and not all_zero_or_one(occupation_numbers):
        raise InvalidParameter(
            f"Invalid initial state specified: instruction={instruction}"
        )

    index = get_fock_space_index(occupation_numbers)

    state._state_vector = connector.assign(state._state_vector, index, coefficient)

    return Result(state=state)


def passive_linear(
    state: PureFockState, instruction: "_PassiveLinearGate", shots: int
) -> Result:
    connector = state._connector
    config = state._config

    modes = instruction.modes

    if config.validate and not are_modes_consecutive(modes):
        raise InvalidParameter(f"Specified modes must be consecutive: modes={modes}")

    unitary = instruction._get_passive_block(connector, config)

    representations = connector.calculate_interferometer_on_fermionic_fock_space(
        unitary, config.cutoff
    )

    state._state_vector = connector.apply_fermionic_passive_linear_to_state_vector(
        representations,
        state._state_vector,
        modes,
        state._d,
        config.cutoff,
    )

    return Result(state=state)


def squeezing2(state: PureFockState, instruction: "Squeezing2", shots: int) -> Result:
    connector = state._connector
    config = state._config

    modes = instruction.modes

    d = state._d
    cutoff = state._config.cutoff

    np = connector.np
    fallback_np = connector.fallback_np

    if config.validate and not are_modes_consecutive(modes):
        raise InvalidParameter(f"Specified modes must be consecutive: modes={modes}")

    r = instruction.params["r"]
    phi = instruction.params["phi"]

    size = get_cutoff_fock_space_dimension(d, cutoff)

    index = fallback_np.zeros(d, dtype=np.int64)

    U = np.array(
        [
            [np.cos(r / 2), np.sin(r / 2) * np.exp(-1j * phi)],
            [-np.sin(r / 2) * np.exp(1j * phi), np.cos(r / 2)],
        ]
    )

    for i in range(size):
        if index[modes[0]] == 0 and index[modes[1]] == 0:
            index[modes[0]] = 1
            index[modes[1]] = 1
            j = get_fock_space_index(index)
            index[modes[0]] = 0
            index[modes[1]] = 0

            if j < size:
                state._state_vector = connector.assign(
                    state._state_vector, ((i, j),), U @ state._state_vector[(i, j),]
                )
            else:
                state._state_vector = connector.assign(
                    state._state_vector, i, U[0, 0] * state._state_vector[i]
                )

        index = next_second_quantized(index)

    return Result(state=state)


def controlled_phase(
    state: PureFockState, instruction: "ControlledPhase", shots: int
) -> Result:
    connector = state._connector

    modes = instruction.modes

    d = state._d
    cutoff = state._config.cutoff

    np = connector.np

    phi = instruction.params["phi"]

    rotation = np.exp(1j * phi)

    indices = calculate_indices_for_controlled_phase(d, cutoff, modes)

    state._state_vector = connector.assign(
        state._state_vector, indices, rotation * state._state_vector[indices]
    )

    return Result(state=state)


def ising_XX(state: PureFockState, instruction: "IsingXX", shots: int) -> Result:
    connector = state._connector
    np = connector.np

    phi = instruction.params["phi"]
    modes = instruction.modes

    d = state._d
    cutoff = state._config.cutoff

    cos_phi = np.cos(phi)
    i_sin_phi = 1j * np.sin(phi)

    indices = calculate_indices_for_ising_XX(d, cutoff, modes)

    for index in indices:
        initial = state._state_vector[index]
        final = cos_phi * initial + i_sin_phi * np.flip(initial)
        state._state_vector = connector.assign(state._state_vector, index, final)

    return Result(state=state)
