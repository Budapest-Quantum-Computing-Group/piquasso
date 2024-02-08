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

from .passive_linear import passive_linear  # noqa: F401

from typing import Optional, Tuple, Mapping

from functools import lru_cache

import random
import numpy as np

from piquasso._math.fock import cutoff_cardinality

from piquasso._math.indices import (
    get_index_in_fock_space,
    get_auxiliary_modes,
)

from ..state import PureFockState
from ..batch_state import BatchPureFockState

from piquasso.instructions import gates

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


def _apply_active_gate_matrix_to_state(
    state: PureFockState,
    matrix: np.ndarray,
    mode: int,
) -> None:
    calculator = state._calculator
    state_vector = state._state_vector
    space = state._space
    auxiliary_subspace = state._get_subspace(state.d - 1)

    @calculator.custom_gradient
    def _apply_active_gate_matrix(state_vector, matrix):
        state_vector = calculator.maybe_convert_to_numpy(state_vector)
        matrix = calculator.maybe_convert_to_numpy(matrix)

        state_index_matrix_list = _calculate_state_index_matrix_list(
            space, auxiliary_subspace, mode
        )
        new_state_vector = _calculate_state_vector_after_apply_active_gate(
            state_vector, matrix, state_index_matrix_list
        )
        grad = _create_linear_active_gate_gradient_function(
            state_vector, matrix, state_index_matrix_list, calculator
        )
        return new_state_vector, grad

    state._state_vector = _apply_active_gate_matrix(state_vector, matrix)


@lru_cache(maxsize=None)
def _calculate_state_index_matrix_list(space, auxiliary_subspace, mode):
    d = space.d
    cutoff = space.cutoff

    state_index_matrix_list = []
    auxiliary_modes = get_auxiliary_modes(d, (mode,))
    all_occupation_numbers = np.zeros(d, dtype=int)

    indices = [cutoff_cardinality(cutoff=n - 1, d=d - 1) for n in range(1, cutoff + 2)]

    for n in range(cutoff):
        limit = space.cutoff - n
        subspace_size = indices[n + 1] - indices[n]

        state_index_matrix = np.empty(shape=(limit, subspace_size), dtype=int)

        for i, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[indices[n] : indices[n + 1]]
        ):
            all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers

            for j in range(limit):
                all_occupation_numbers[mode] = j
                state_index_matrix[j, i] = get_index_in_fock_space(
                    tuple(all_occupation_numbers)
                )

        state_index_matrix_list.append(state_index_matrix)

    return state_index_matrix_list


def _calculate_state_vector_after_apply_active_gate(
    state_vector, matrix, state_index_matrix_list
):
    new_state_vector = np.empty_like(state_vector, dtype=state_vector.dtype)

    is_batch = len(state_vector.shape) == 2

    einsum_string = "ij,jkl->ikl" if is_batch else "ij,jk->ik"

    for state_index_matrix in state_index_matrix_list:
        limit = state_index_matrix.shape[0]
        new_state_vector[state_index_matrix] = np.einsum(
            einsum_string, matrix[:limit, :limit], state_vector[state_index_matrix]
        )

    return new_state_vector


def _create_linear_active_gate_gradient_function(
    state_vector,
    matrix,
    state_index_matrix_list,
    calculator,
):
    def linear_active_gate_gradient(upstream):
        tf = calculator._tf
        fallback_np = calculator.fallback_np

        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            np = calculator.fallback_np
            upstream = upstream.numpy()
        else:
            np = calculator.np

        cutoff = len(matrix)

        is_batch = len(state_vector.shape) == 2

        matrix_einsum_string = "ijl,kjl->ki" if is_batch else "ij,kj->ki"
        initial_state_einsum_string = "ji,jkl->ikl" if is_batch else "ji,jk->ik"

        reshape_arg = (-1, state_vector.shape[1]) if is_batch else (-1,)

        unordered_gradient_by_initial_state = []
        order_by = []

        gradient_by_matrix = np.zeros(shape=(cutoff, cutoff), dtype=state_vector.dtype)

        conjugated_matrix = np.conj(matrix)
        conjugated_state_vector = np.conj(state_vector)
        for indices in state_index_matrix_list:
            limit = indices.shape[0]

            upstream_slice = upstream[indices]
            state_vector_slice = conjugated_state_vector[indices]

            matrix_slice = conjugated_matrix[:limit, :limit]

            order_by.append(indices.reshape(-1))
            product = np.einsum(
                initial_state_einsum_string, matrix_slice, upstream_slice
            )
            unordered_gradient_by_initial_state.append(product.reshape(*reshape_arg))

            partial_gradient = np.einsum(
                matrix_einsum_string, state_vector_slice, upstream_slice
            )

            if static_valued:
                gradient_by_matrix[:limit, :limit] += partial_gradient
            else:
                gradient_by_matrix += np.pad(
                    partial_gradient,
                    [[0, cutoff - limit], [0, cutoff - limit]],
                    "constant",
                )

        gradient_by_initial_state = np.concatenate(unordered_gradient_by_initial_state)[
            fallback_np.concatenate(order_by).argsort()
        ]

        if static_valued:
            return (
                tf.constant(gradient_by_initial_state),
                tf.constant(gradient_by_matrix),
            )

        return gradient_by_initial_state, gradient_by_matrix

    return linear_active_gate_gradient


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
    xi = instruction._all_params["xi"]
    np = state._np

    mode = instruction.modes[0]

    coefficients = np.exp(
        1j * xi * np.array([basis[mode] ** 2 for basis in state._space])
    )

    # NOTE: Transposition is done here in order to work with batch processing.
    state._state_vector = (coefficients * state._state_vector.T).T

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
    r = instruction._all_params["r"]
    phi = instruction._all_params["phi"]

    matrix = state._space.get_single_mode_displacement_operator(r=r, phi=phi)

    _apply_active_gate_matrix_to_state(state, matrix, instruction.modes[0])

    state.normalize()

    return Result(state=state)


def squeezing(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    matrix = state._space.get_single_mode_squeezing_operator(
        r=instruction._all_params["r"],
        phi=instruction._all_params["phi"],
    )

    _apply_active_gate_matrix_to_state(state, matrix, instruction.modes[0])

    state.normalize()

    return Result(state=state)


def cubic_phase(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    gamma = instruction._all_params["gamma"]
    hbar = state._config.hbar

    matrix = state._space.get_single_mode_cubic_phase_operator(
        gamma=gamma, hbar=hbar, calculator=state._calculator
    )
    _apply_active_gate_matrix_to_state(state, matrix, instruction.modes[0])

    state.normalize()

    return Result(state=state)


def linear(
    state: PureFockState, instruction: gates._ActiveLinearGate, shots: int
) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        passive_block=instruction._get_passive_block(state._calculator, state._config),
        active_block=instruction._get_active_block(state._calculator, state._config),
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
    modes: Optional[Tuple[int, ...]] = None,
) -> None:
    if modes:
        occupation_numbers = state._space.get_occupied_basis(
            modes=modes, occupation_numbers=occupation_numbers
        )

    index = state._space.index(occupation_numbers)

    state._state_vector = state._calculator.assign(
        state._state_vector, index, coefficient
    )


def batch_prepare(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    subprograms = instruction._all_params["subprograms"]
    execute = instruction._all_params["execute"]

    state_vectors = [
        execute(subprogram, shots).state._state_vector for subprogram in subprograms
    ]

    batch_state = BatchPureFockState(
        d=state.d, calculator=state._calculator, config=state._config
    )

    batch_state._apply_separate_state_vectors(state_vectors)

    return Result(state=batch_state)


def batch_apply(
    state: BatchPureFockState, instruction: Instruction, shots: int
) -> Result:
    subprograms = instruction._all_params["subprograms"]
    execute = instruction._all_params["execute"]

    d = state.d
    calculator = state._calculator
    config = state._config

    resulting_state_vectors = []

    for state_vector, subprogram in zip(state._batch_state_vectors, subprograms):
        small_state = PureFockState(d=d, calculator=calculator, config=config)

        small_state._state_vector = state_vector

        resulting_state_vectors.append(
            execute(subprogram, initial_state=small_state).state._state_vector
        )

    state._apply_separate_state_vectors(resulting_state_vectors)

    return Result(state=state)
