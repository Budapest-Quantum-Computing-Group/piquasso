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

from typing import Tuple, Mapping, Dict

import random
import numpy as np

from .state import FockState

from piquasso._math.fock import FockBasis
from piquasso.api.instruction import Instruction
from piquasso.api.result import Result


def vacuum(state: FockState, instruction: Instruction) -> FockState:
    state.reset()

    return state


def passive_linear(state: FockState, instruction: Instruction) -> FockState:
    operator: np.ndarray = instruction._all_params["passive_block"]

    index = state._get_operator_index(instruction.modes)

    embedded_operator = np.identity(state._space.d, dtype=complex)

    embedded_operator[index] = operator

    fock_operator = state._space.get_passive_fock_operator(embedded_operator)

    state._density_matrix = (
        fock_operator @ state._density_matrix @ fock_operator.conjugate().transpose()
    )

    return state


def particle_number_measurement(
    state: FockState, instruction: Instruction
) -> FockState:
    probability_map = _get_probability_map(
        state=state,
        modes=instruction.modes,
    )

    samples = random.choices(
        population=list(probability_map.keys()),
        weights=list(probability_map.values()),
        k=state.shots,
    )

    # NOTE: We choose the last sample for multiple shots.
    sample = samples[-1]

    normalization = _get_normalization(probability_map, sample)

    _project_to_subspace(
        state=state,
        subspace_basis=sample,
        modes=instruction.modes,
        normalization=normalization,
    )

    state.result = Result(samples=samples)  # type: ignore

    return state


def _get_probability_map(
    state: FockState, *, modes: Tuple[int, ...]
) -> Dict[FockBasis, float]:
    probability_map: Dict[FockBasis, float] = {}

    for index, basis in state._space.operator_basis_diagonal_on_modes(modes=modes):
        coefficient = float(state._density_matrix[index])

        subspace_basis = basis.ket.on_modes(modes=modes)

        if subspace_basis in probability_map:
            probability_map[subspace_basis] += coefficient
        else:
            probability_map[subspace_basis] = coefficient

    return probability_map


def _get_normalization(
    probability_map: Mapping[FockBasis, float], sample: FockBasis
) -> float:
    return 1 / probability_map[sample]


def _project_to_subspace(
    state: FockState,
    *,
    subspace_basis: FockBasis,
    modes: Tuple[int, ...],
    normalization: float
) -> None:
    projected_density_matrix = _get_projected_density_matrix(
        state=state,
        subspace_basis=subspace_basis,
        modes=modes,
    )

    state._density_matrix = projected_density_matrix * normalization


def _get_projected_density_matrix(
    state: FockState, *, subspace_basis: FockBasis, modes: Tuple[int, ...]
) -> np.ndarray:
    new_density_matrix = state._get_empty()

    index = state._space.get_projection_operator_indices(
        subspace_basis=subspace_basis,
        modes=modes,
    )

    new_density_matrix[index] = state._density_matrix[index]

    return new_density_matrix


def create(state: FockState, instruction: Instruction) -> FockState:
    operator = state._space.get_creation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return state


def annihilate(state: FockState, instruction: Instruction) -> FockState:
    operator = state._space.get_annihilation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return state


def kerr(state: FockState, instruction: Instruction) -> FockState:
    mode = instruction.modes[0]
    xi = instruction._all_params["xi"]

    for index, (basis, dual_basis) in state._space.operator_basis:
        number = basis[mode]
        dual_number = dual_basis[mode]

        coefficient = np.exp(
            1j * xi * (number * (2 * number + 1) - dual_number * (2 * dual_number + 1))
        )

        state._density_matrix[index] *= coefficient

    return state


def cross_kerr(state: FockState, instruction: Instruction) -> FockState:
    modes = instruction.modes
    xi = instruction._all_params["xi"]

    for index, (basis, dual_basis) in state._space.operator_basis:
        coefficient = np.exp(
            1j
            * xi
            * (
                basis[modes[0]] * basis[modes[1]]
                - dual_basis[modes[0]] * dual_basis[modes[1]]
            )
        )

        state._density_matrix[index] *= coefficient

    return state


def linear(state: FockState, instruction: Instruction) -> FockState:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        cache_size=state._config.cache_size,
        auxiliary_modes=state._get_auxiliary_modes(instruction.modes),
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        displacement=instruction._all_params["displacement_vector"],
    )

    state._density_matrix = (
        operator @ state._density_matrix @ operator.conjugate().transpose()
    )

    state.normalize()

    return state


def density_matrix_instruction(state: FockState, instruction: Instruction) -> FockState:
    _add_occupation_number_basis(state, **instruction.params)

    return state


def _add_occupation_number_basis(
    state: FockState,
    *,
    ket: Tuple[int, ...],
    bra: Tuple[int, ...],
    coefficient: complex
) -> None:
    index = state._space.index(ket)
    dual_index = state._space.index(bra)

    state._density_matrix[index, dual_index] = coefficient
