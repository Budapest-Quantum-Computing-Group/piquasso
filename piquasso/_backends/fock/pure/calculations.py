#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from piquasso._math.fock import FockSpace, cutoff_cardinality

from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_subspace,
)

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


def _get_interferometer_on_fock_space(interferometer, space, calculator):
    """Calculates finite representation of interferometer in the Fock space.

    The function assumes the knowledge of the 1-particle unitary.

    Sources:
    - Fast optimization of parametrized quantum optical circuits
    (https://quantum-journal.org/papers/q-2020-11-30-366/)

    Args:
        interferometer (numpy.ndarray): The 1-particle unitary
        space (FockSpace): List of basis elements on the Fock space.
        calculator (BaseCalculator): Object containing calculations.

    Returns:
        numpy.ndarray: Finite representation of interferometer in the Fock space
    """

    np = calculator.np
    true_np = calculator.fallback_np

    cutoff = space.cutoff
    d = space.d

    space = [true_np.array(element, dtype=int) for element in space]

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)

    indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]

        new_subspace_representation = np.zeros(shape=(size,) * 2, dtype=complex)
        previous_representation = subspace_representations[n - 1]

        subspace = space[indices[n - 1] : indices[n]]

        update_indices = []
        updates = []

        for row_index, row_vector in enumerate(subspace):
            first_nonzero_row_index = true_np.nonzero(row_vector)[0][0]
            row_occupation_number = row_vector[first_nonzero_row_index]

            row_vector[first_nonzero_row_index] -= 1
            subspace_row_index = get_index_in_fock_subspace(tuple(row_vector))
            row_vector[first_nonzero_row_index] += 1

            for column_index, column_vector in enumerate(subspace):

                nonzero_column_indices = true_np.nonzero(column_vector)[0]
                subspace_column_indices = []

                for nonzero_column_index in nonzero_column_indices:
                    column_vector[nonzero_column_index] -= 1
                    subspace_column_indices.append(
                        get_index_in_fock_subspace(tuple(column_vector))
                    )
                    column_vector[nonzero_column_index] += 1

                update_indices.append([row_index, column_index])
                updates.append(
                    (
                        np.sqrt(column_vector[(nonzero_column_indices,)])
                        * interferometer[
                            first_nonzero_row_index, nonzero_column_indices
                        ]
                    )
                    @ previous_representation[
                        subspace_row_index, subspace_column_indices
                    ]
                    / np.sqrt(row_occupation_number)
                )

        new_subspace_representation = calculator.scatter(update_indices, updates, size)

        subspace_representations.append(
            new_subspace_representation.astype(interferometer.dtype)
        )

    return calculator.block_diag(*subspace_representations)


def passive_linear(
    state: PureFockState, instruction: Instruction, shots: int
) -> Result:
    calculator = state._calculator

    interferometer: np.ndarray = instruction._all_params["passive_block"]

    subspace = FockSpace(
        d=len(interferometer), cutoff=state._space.cutoff, calculator=calculator
    )

    interferometer_on_fock_space = _get_interferometer_on_fock_space(
        interferometer, subspace, calculator
    )

    _apply_subspace_matrix_to_state(
        state, interferometer_on_fock_space, instruction.modes
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


def _apply_subspace_matrix_to_state(
    state: PureFockState,
    matrix: np.ndarray,
    modes: Tuple[int, ...],
) -> None:
    np = state._np
    fallback_np = state._calculator.fallback_np

    old_state_vector = state._state_vector
    new_state_vector = []

    space = state._space
    subspace = FockSpace(
        d=len(modes), cutoff=state._space.cutoff, calculator=state._calculator
    )

    for row_vector in state._space:
        row_vector_array = fallback_np.array(row_vector, dtype=int)
        row_index_on_subspace = get_index_in_fock_space(
            tuple(row_vector_array[(modes,)])
        )
        row_vector_array[(modes,)] = [0] * len(modes)

        matrix_row = matrix[row_index_on_subspace, :]

        matrix_column_indices = []
        state_indices = []

        limit = cutoff_cardinality(
            cutoff=(space.cutoff - sum(row_vector_array)), d=len(modes)
        )

        for column_index_on_subspace, column_vector_on_subspace in enumerate(
            subspace[:limit]
        ):
            row_vector_array[(modes,)] = column_vector_on_subspace
            column_index = get_index_in_fock_space(tuple(row_vector_array))
            matrix_column_indices.append(column_index_on_subspace)
            state_indices.append(column_index)

        new_state_vector.append(
            np.dot(
                matrix_row[matrix_column_indices],
                old_state_vector[state_indices],
            )
        )

    state._state_vector = np.stack(new_state_vector)


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
    xi_vector = instruction._all_params["xi_vector"]

    for mode_index, mode in enumerate(instruction.modes):
        xi = xi_vector[mode_index]

        for index, basis in state._space.basis:
            number = basis[mode]
            coefficient = state._np.exp(1j * xi * number * (2 * number + 1))
            state._state_vector = state._calculator.assign(
                state._state_vector, index, state._state_vector[index] * coefficient
            )

    return Result(state=state)


def cross_kerr(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    np = state._calculator.np

    modes = instruction.modes
    xi = instruction._all_params["xi"]
    for index, basis in state._space.basis:
        coefficient = np.exp(1j * xi * basis[modes[0]] * basis[modes[1]])
        state._state_vector = state._calculator.assign(
            state._state_vector, index, state._state_vector[index] * coefficient
        )

    return Result(state=state)


def displacement(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    np = state._np

    amplitudes = np.abs(instruction._all_params["displacement_vector"])
    angles = np.angle(instruction._all_params["displacement_vector"])

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_displacement_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        _apply_subspace_matrix_to_state(
            state,
            matrix,
            (mode,),
        )

        state.normalize()

    return Result(state=state)


def squeezing(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    np = state._np

    amplitudes = np.arccosh(np.diag(instruction._all_params["passive_block"]))
    angles = np.angle(-np.diag(instruction._all_params["active_block"]))

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_squeezing_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        _apply_subspace_matrix_to_state(
            state,
            matrix,
            (mode,),
        )

        state.normalize()

    return Result(state=state)


def cubic_phase(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    gamma_vector = instruction._all_params["gamma_vector"]
    hbar = state._config.hbar

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_cubic_phase_operator(
            gamma=gamma_vector[index], hbar=hbar, calculator=state._calculator
        )
        _apply_subspace_matrix_to_state(
            state,
            matrix,
            (mode,),
        )

        state.normalize()

    return Result(state=state)


def linear(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
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

    state._state_vector = state._calculator.assign(
        state._state_vector, index, coefficient
    )
