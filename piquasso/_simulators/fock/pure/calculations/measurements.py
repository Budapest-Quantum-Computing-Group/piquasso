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

from piquasso._math.fock import get_fock_space_basis
from piquasso._simulators.fock.calculations import get_projection_operator_indices
from piquasso._simulators.fock.general.state import FockState

from ..state import PureFockState


def post_select_photons(
    state: PureFockState, instruction: PostSelectPhotons, shots: int
) -> Result:
    connector = state._connector

    postselect_modes = instruction.params["postselect_modes"]

    photon_counts = instruction.params["photon_counts"]

    index = get_projection_operator_indices(
        d=state.d,
        cutoff=state._config.cutoff,
        modes=postselect_modes,
        basis_vector=photon_counts,
    )
    small_index = connector.fallback_np.arange(index.shape[0])

    new_state = PureFockState(
        d=state.d - len(postselect_modes),
        connector=state._connector,
        config=state._config,
    )

    new_state.state_vector = connector.assign(
        new_state.state_vector, small_index, state.state_vector[index]
    )

    return Result(state=new_state)


def imperfect_post_select_photons(
    state: PureFockState, instruction: PostSelectPhotons, shots: int
) -> Result:
    np = state._connector.np

    postselect_modes = instruction.params["postselect_modes"]

    photon_counts = instruction.params["photon_counts"]

    detector_efficiency_matrix = instruction.params["detector_efficiency_matrix"]

    postselect_basis = get_fock_space_basis(
        d=len(postselect_modes), cutoff=len(detector_efficiency_matrix)
    )

    new_state = FockState(
        d=state.d - len(postselect_modes),
        connector=state._connector,
        config=state._config,
    )

    for occupation_numbers in postselect_basis:
        detector_probability = np.prod(
            detector_efficiency_matrix[(photon_counts, occupation_numbers)]
        )

        index = get_projection_operator_indices(
            state.d, state._config.cutoff, postselect_modes, occupation_numbers
        )

        small_index = np.arange(index.shape[0])

        state_vector = np.zeros(
            shape=new_state.density_matrix.shape[0],
            dtype=new_state.density_matrix.dtype,
        )

        state_vector = state._connector.assign(
            state_vector, small_index, state.state_vector[index]
        )

        new_state._density_matrix += detector_probability * np.outer(
            state_vector, np.conj(state_vector)
        )

    return Result(state=new_state)
