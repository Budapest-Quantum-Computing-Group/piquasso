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

from typing import Tuple, Dict, Mapping

import random
import numpy as np

from .state import PureFockState

from piquasso.api.result import Result
from piquasso.api.instruction import Instruction

from piquasso._math.fock import FockBasis


def particle_number_measurement(
    state: PureFockState, instruction: Instruction
) -> Result:
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

    return Result(state=state, samples=samples)  # type: ignore


def vacuum(state: PureFockState, instruction: Instruction) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(state: PureFockState, instruction: Instruction) -> Result:
    operator: np.ndarray = instruction._all_params["passive_block"]

    index = state._get_operator_index(instruction.modes)

    embedded_operator = np.identity(state._space.d, dtype=complex)

    embedded_operator[index] = operator

    fock_operator = state._space.get_passive_fock_operator(embedded_operator)

    state._state_vector = fock_operator @ state._state_vector

    return Result(state=state)


def _get_probability_map(
    state: PureFockState, *, modes: Tuple[int, ...]
) -> Dict[FockBasis, float]:
    probability_map: Dict[FockBasis, float] = {}

    for index, basis in state._space.basis:
        coefficient = float(state._state_vector[index])

        subspace_basis = basis.on_modes(modes=modes)

        if subspace_basis in probability_map:
            probability_map[subspace_basis] += coefficient ** 2
        else:
            probability_map[subspace_basis] = coefficient ** 2

    return probability_map


def _get_normalization(
    probability_map: Mapping[FockBasis, float], sample: FockBasis
) -> float:
    return np.sqrt(1 / probability_map[sample])


def _project_to_subspace(
    state: PureFockState,
    *,
    subspace_basis: FockBasis,
    modes: Tuple[int, ...],
    normalization: float
) -> None:
    projected_state_vector = _get_projected_state_vector(
        state=state,
        subspace_basis=subspace_basis,
        modes=modes,
    )

    state._state_vector = projected_state_vector * normalization


def _get_projected_state_vector(
    state: PureFockState, *, subspace_basis: FockBasis, modes: Tuple[int, ...]
) -> np.ndarray:
    new_state_vector = state._get_empty()

    index = state._space.get_projection_operator_indices_for_pure(
        subspace_basis=subspace_basis,
        modes=modes,
    )

    new_state_vector[index] = state._state_vector[index]

    return new_state_vector


def create(state: PureFockState, instruction: Instruction) -> Result:
    operator = state._space.get_creation_operator(instruction.modes)

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def annihilate(state: PureFockState, instruction: Instruction) -> Result:
    operator = state._space.get_annihilation_operator(instruction.modes)

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def kerr(state: PureFockState, instruction: Instruction) -> Result:
    mode = instruction.modes[0]
    xi = instruction._all_params["xi"]

    for index, basis in state._space.basis:
        number = basis[mode]
        coefficient = np.exp(1j * xi * number * (2 * number + 1))
        state._state_vector[index] *= coefficient

    return Result(state=state)


def cross_kerr(state: PureFockState, instruction: Instruction) -> Result:
    modes = instruction.modes
    xi = instruction._all_params["xi"]

    for index, basis in state._space.basis:
        coefficient = np.exp(1j * xi * basis[modes[0]] * basis[modes[1]])
        state._state_vector[index] *= coefficient

    return Result(state=state)


def linear(state: PureFockState, instruction: Instruction) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        cache_size=state._config.cache_size,
        auxiliary_modes=state._get_auxiliary_modes(instruction.modes),
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        displacement=instruction._all_params["displacement_vector"],
    )

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def state_vector_instruction(state: PureFockState, instruction: Instruction) -> Result:
    _add_occupation_number_basis(
        state=state,
        **instruction._all_params,
        modes=instruction.modes,
    )

    return Result(state=state)


def _add_occupation_number_basis(  # type: ignore
    state: PureFockState,
    coefficient: complex,
    occupation_numbers: Tuple[int, ...],
    modes: Tuple[int, ...] = None,
) -> None:
    if modes:
        occupation_numbers = state._space.get_occupied_basis(
            modes=modes, occupation_numbers=occupation_numbers
        )

    index = state._space.index(occupation_numbers)

    state._state_vector[index] = coefficient
