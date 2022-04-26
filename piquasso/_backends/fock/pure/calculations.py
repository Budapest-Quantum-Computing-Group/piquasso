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

from scipy.linalg import block_diag

from .state import PureFockState

from piquasso.api.result import Result
from piquasso.api.instruction import Instruction


def particle_number_measurement(
    state: PureFockState, instruction: Instruction, shots: int
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


def vacuum(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(
    state: PureFockState, instruction: Instruction, shots: int
) -> Result:
    matrix: np.ndarray = instruction._all_params["passive_block"]

    matrix_on_fock_space = block_diag(
        *(
            state._space.symmetric_tensorpower(
                matrix,
                n,
                permanent_function=state._config.permanent_function,
            )
            for n in range(state._space.cutoff)
        )
    )

    modes = instruction.modes

    _apply_subsystem_representation_to_state(
        state, matrix_on_fock_space, modes, state._get_auxiliary_modes(modes)
    )

    return Result(state=state)


def _get_normalization(
    probability_map: Mapping[Tuple[int, ...], float], sample: Tuple[int, ...]
) -> float:
    return np.sqrt(1 / probability_map[sample])


def _project_to_subspace(
    state: PureFockState,
    *,
    subspace_basis: Tuple[int, ...],
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
    state: PureFockState, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
) -> np.ndarray:
    new_state_vector = state._get_empty()

    index = state._space.get_projection_operator_indices_for_pure(
        subspace_basis=subspace_basis,
        modes=modes,
    )

    new_state_vector[index] = state._state_vector[index]

    return new_state_vector


def _apply_subsystem_representation_to_state(
    state: PureFockState,
    matrix: np.ndarray,
    modes: Tuple[int, ...],
    auxiliary_modes: Tuple[int, ...],
) -> None:
    new_state_vector = np.zeros_like(state._state_vector)

    from piquasso._math.fock import FockSpace

    subsystem_space = FockSpace(d=len(modes), cutoff=state._space.cutoff)

    for multimode_index, multimode_vector in enumerate(state._space):
        single_mode_basis_index = subsystem_space.index(
            multimode_vector.on_modes(modes=modes)
        )

        for running_vector in state._space:
            if multimode_vector.on_modes(
                modes=auxiliary_modes
            ) == running_vector.on_modes(modes=auxiliary_modes):

                single_mode_running_index = subsystem_space.index(
                    running_vector.on_modes(modes=modes)
                )

                index_on_multimode = state._space.index(running_vector)

                single_mode_matrix_index = (
                    single_mode_basis_index,
                    single_mode_running_index,
                )

                new_state_vector[multimode_index] += (
                    matrix[single_mode_matrix_index]
                    * state._state_vector[index_on_multimode]
                )

    state._state_vector = new_state_vector


def create(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_creation_operator(instruction.modes)

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def annihilate(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_annihilation_operator(instruction.modes)

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def kerr(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    xi_vector = instruction._all_params["xi"]

    for mode_index, mode in enumerate(instruction.modes):
        xi = xi_vector[mode_index]

        for index, basis in state._space.basis:
            number = basis[mode]
            coefficient = np.exp(1j * xi.squeeze() * number * (2 * number + 1))
            state._state_vector[index] *= coefficient

    return Result(state=state)


def cross_kerr(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    modes = instruction.modes
    xi = instruction._all_params["xi"]
    for index, basis in state._space.basis:
        coefficient = np.exp(1j * xi * basis[modes[0]] * basis[modes[1]])
        state._state_vector[index] *= coefficient

    return Result(state=state)


def displacement(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.abs(instruction._all_params["displacement_vector"])
    angles = np.angle(instruction._all_params["displacement_vector"])

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_displacement_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        _apply_subsystem_representation_to_state(
            state, matrix, (mode,), state._get_auxiliary_modes((mode,))
        )

        state.normalize()

    return Result(state=state)


def squeezing(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.arccosh(np.diag(instruction._all_params["passive_block"]))
    angles = np.angle(-np.diag(instruction._all_params["active_block"]))

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_squeezing_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        _apply_subsystem_representation_to_state(
            state, matrix, (mode,), state._get_auxiliary_modes((mode,))
        )

        state.normalize()

    return Result(state=state)


def cubic_phase(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    gamma = instruction._all_params["gamma"]
    hbar = state._config.hbar

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_cubic_phase_operator(
            gamma=gamma[index], hbar=hbar
        )
        _apply_subsystem_representation_to_state(
            state, matrix, (mode,), state._get_auxiliary_modes((mode,))
        )

        state.normalize()

    return Result(state=state)


def linear(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        permanent_function=state._config.permanent_function,
    )

    state._state_vector = operator @ state._state_vector

    state.normalize()

    return Result(state=state)


def state_vector_instruction(
    state: PureFockState, instruction: Instruction, shots: int
) -> Result:
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
