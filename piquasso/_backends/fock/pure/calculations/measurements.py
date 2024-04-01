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

from piquasso.instructions.measurements import PostSelectPhotons
from piquasso.api.result import Result


from ...calculations import get_projection_operator_indices

from ..state import PureFockState


def post_select_photons(
    state: PureFockState, instruction: PostSelectPhotons, shots: int
) -> Result:
    calculator = state._calculator

    postselect_modes = instruction.params["postselect_modes"]

    photon_counts = instruction.params["photon_counts"]

    index = get_projection_operator_indices(
        d=state.d,
        cutoff=state._config.cutoff,
        modes=postselect_modes,
        basis_vector=photon_counts,
    )
    small_index = calculator.fallback_np.arange(index.shape[0])

    new_state = PureFockState(
        d=state.d - len(postselect_modes),
        calculator=state._calculator,
        config=state._config,
    )

    new_state.state_vector = calculator.assign(
        new_state.state_vector, small_index, state.state_vector[index]
    )

    return Result(state=new_state)
