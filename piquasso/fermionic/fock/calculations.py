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

from piquasso.api.result import Result

from piquasso.api.exceptions import InvalidParameter

from piquasso._math.validations import all_zero_or_one, are_modes_consecutive

from .._utils import get_fock_space_index
from .state import PureFockState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piquasso.instructions.preparations import StateVector
    from piquasso.instructions.gates import _PassiveLinearGate


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
