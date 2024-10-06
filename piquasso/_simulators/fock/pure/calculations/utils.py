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

from ...calculations import get_projection_operator_indices

from ..state import PureFockState

from piquasso.api.connector import BaseConnector


def project_to_subspace(
    state: PureFockState,
    *,
    subspace_basis: Tuple[int, ...],
    modes: Tuple[int, ...],
    normalization: float,
    connector: BaseConnector,
) -> None:
    projected_state_vector = _get_projected_state_vector(
        state=state, subspace_basis=subspace_basis, modes=modes, connector=connector
    )

    state.state_vector = projected_state_vector * normalization


def _get_projected_state_vector(
    state: PureFockState,
    *,
    subspace_basis: Tuple[int, ...],
    modes: Tuple[int, ...],
    connector: BaseConnector,
) -> np.ndarray:
    new_state_vector = state._get_empty()

    index = get_projection_operator_indices(
        state.d,
        state._config.cutoff,
        modes,
        subspace_basis,
    )

    new_state_vector = connector.assign(
        new_state_vector, index, state.state_vector[index]
    )

    return new_state_vector
