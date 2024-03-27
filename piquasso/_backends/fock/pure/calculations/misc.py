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

from piquasso.instructions.misc import PostSelectPhotons
from piquasso.api.result import Result

from piquasso._math.fock import get_fock_space_basis
from piquasso._backends.fock.general.state import FockState

from .utils import project_to_subspace, get_projected_state_vector

from ..state import PureFockState


def post_select_photons(
    state: PureFockState, instruction: PostSelectPhotons, shots: int
) -> Result:
    project_to_subspace(
        state,
        subspace_basis=instruction.params["photon_counts"],
        modes=instruction.params["postselect_modes"],
        normalization=1.0,
    )

    state.normalize()

    return Result(state=state)


def imperfect_post_select_photons(
    state: PureFockState, instruction: PostSelectPhotons, shots: int
) -> Result:
    np = state._calculator.np

    state_vector_size = len(state.state_vector)

    postselect_modes = instruction.params["postselect_modes"]

    photon_counts = instruction.params["photon_counts"]

    detector_efficiency_matrix = instruction.params["detector_efficiency_matrix"]

    postselect_basis = get_fock_space_basis(
        d=len(postselect_modes), cutoff=len(detector_efficiency_matrix)
    )
    density_matrix = np.zeros(
        shape=(state_vector_size, state_vector_size), dtype=state.state_vector.dtype
    )

    for occupation_numbers in postselect_basis:
        detector_probability = 1.0
        for j, count in enumerate(photon_counts):
            detector_probability *= detector_efficiency_matrix[
                count, occupation_numbers[j]
            ]

        particle_detection_probability = (
            state.get_particle_detection_probability_on_modes(
                occupation_numbers, postselect_modes
            )
        )

        probability = detector_probability * particle_detection_probability

        state_vector = get_projected_state_vector(
            state,
            subspace_basis=occupation_numbers,
            modes=postselect_modes,
        )

        density_matrix += probability * np.outer(state_vector.conj(), state_vector)

    normalized_density_matrix = density_matrix / np.trace(density_matrix)

    new_state = FockState(d=state.d, calculator=state._calculator, config=state._config)

    new_state._density_matrix = normalized_density_matrix

    return Result(state=new_state)
