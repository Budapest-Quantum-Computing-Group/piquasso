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

from typing import Tuple

import numpy as np

from ...calculations import get_projection_operator_indices

from ..state import PureFockState


def project_to_subspace(
    state: PureFockState,
    *,
    subspace_basis: Tuple[int, ...],
    modes: Tuple[int, ...],
    normalization: float,
) -> PureFockState:
    remaining_state_vector = normalization * _get_remaining_state_vector(
        state=state,
        subspace_basis=subspace_basis,
        modes=modes,
    )

    config_copy = state._config.copy()

    config_copy.cutoff -= sum(subspace_basis)

    new_state = PureFockState(
        d=state.d - len(subspace_basis), connector=state._connector, config=config_copy
    )

    new_state.state_vector = remaining_state_vector

    return new_state


def _get_remaining_state_vector(
    state: PureFockState,
    subspace_basis: Tuple[int, ...],
    modes: Tuple[int, ...],
) -> np.ndarray:
    index = get_projection_operator_indices(
        state.d,
        state._config.cutoff,
        modes,
        subspace_basis,
    )

    return state.state_vector[index]
