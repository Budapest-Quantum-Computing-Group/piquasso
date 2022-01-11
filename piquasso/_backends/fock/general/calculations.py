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

from typing import Tuple, Mapping

import random
import numpy as np

from .state import FockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result


def vacuum(state: FockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator: np.ndarray = instruction._all_params["passive_block"]

    fock_operator = state._space.get_passive_fock_operator(
        operator,
        modes=instruction.modes,
        d=state._space.d,
        permanent_function=state._config.permanent_function,
    )

    state._density_matrix = (
        fock_operator @ state._density_matrix @ fock_operator.conjugate().transpose()
    )

    return Result(state=state)


def particle_number_measurement(
    state: FockState, instruction: Instruction, shots: int
) -> Result:

    reduced_state = state.reduced(instruction.modes)

    probability_map = reduced_state.fock_probabilities_map

    samples = random.choices(
        population=list(probability_map.keys()),
        weights=list(probability_map.values()),
        k=shots,
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

    return Result(state=state, samples=samples)


def _get_normalization(
    probability_map: Mapping[Tuple[int, ...], float], sample: Tuple[int, ...]
) -> float:
    return 1 / probability_map[sample]


def _project_to_subspace(
    state: FockState,
    *,
    subspace_basis: Tuple[int, ...],
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
    state: FockState, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
) -> np.ndarray:
    new_density_matrix = state._get_empty()

    index = state._space.get_projection_operator_indices(
        subspace_basis=subspace_basis,
        modes=modes,
    )

    new_density_matrix[index] = state._density_matrix[index]

    return new_density_matrix


def create(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_creation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return Result(state=state)


def annihilate(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_annihilation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return Result(state=state)


def kerr(state: FockState, instruction: Instruction, shots: int) -> Result:
    mode = instruction.modes[0]
    xi = instruction._all_params["xi"]

    for index, (basis, dual_basis) in state._space.operator_basis:
        number = basis[mode]
        dual_number = dual_basis[mode]

        coefficient = np.exp(
            1j * xi * (number * (2 * number + 1) - dual_number * (2 * dual_number + 1))
        )

        state._density_matrix[index] *= coefficient

    return Result(state=state)


def cross_kerr(state: FockState, instruction: Instruction, shots: int) -> Result:
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

    return Result(state=state)


def displacement(state: FockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.abs(instruction._all_params["displacement_vector"])
    angles = np.angle(instruction._all_params["displacement_vector"])

    for index, mode in enumerate(instruction.modes):
        operator = state._space.get_single_mode_displacement_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        embedded_operator = state._space.embed_matrix(
            operator,
            modes=(mode,),
            auxiliary_modes=state._get_auxiliary_modes(instruction.modes),
        )

        state._density_matrix = (
            embedded_operator
            @ state._density_matrix
            @ embedded_operator.conjugate().transpose()
        )

        state.normalize()

    return Result(state=state)


def squeezing(state: FockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.arccosh(np.diag(instruction._all_params["passive_block"]))
    angles = np.angle(-np.diag(instruction._all_params["active_block"]))

    for index, mode in enumerate(instruction.modes):
        operator = state._space.get_single_mode_squeezing_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        embedded_operator = state._space.embed_matrix(
            operator,
            modes=(mode,),
            auxiliary_modes=state._get_auxiliary_modes(instruction.modes),
        )

        state._density_matrix = (
            embedded_operator
            @ state._density_matrix
            @ embedded_operator.conjugate().transpose()
        )

        state.normalize()

    return Result(state=state)


def linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        permanent_function=state._config.permanent_function,
    )

    state._density_matrix = (
        operator @ state._density_matrix @ operator.conjugate().transpose()
    )

    state.normalize()

    return Result(state=state)


def density_matrix_instruction(
    state: FockState, instruction: Instruction, shots: int
) -> Result:
    _add_occupation_number_basis(state, **instruction.params)

    return Result(state=state)


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
