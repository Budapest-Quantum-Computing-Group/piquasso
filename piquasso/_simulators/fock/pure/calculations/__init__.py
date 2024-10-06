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

from .passive_linear import passive_linear, beamsplitter5050
from .measurements import (
    post_select_photons,
    imperfect_post_select_photons,
)

from .homodyne import homodyne_measurement

__all__ = [
    "passive_linear",
    "beamsplitter5050",
    "post_select_photons",
    "imperfect_post_select_photons",
    "homodyne_measurement",
]

from typing import Optional, Tuple, Mapping

import random
import numpy as np

from .passive_linear import _apply_passive_linear
from .utils import project_to_subspace

from ...calculations import calculate_state_index_matrix_list

from ..state import PureFockState
from ..batch_state import BatchPureFockState

from piquasso._math.decompositions import euler
from piquasso._math.indices import get_index_in_fock_space
from piquasso._math.fock import (
    get_single_mode_displacement_operator,
    get_creation_operator,
    get_annihilation_operator,
    get_single_mode_squeezing_operator,
    get_single_mode_cubic_phase_operator,
    get_fock_space_basis,
)

from piquasso.instructions import gates

from piquasso.api.result import Result
from piquasso.api.instruction import Instruction
from piquasso.api.connector import BaseConnector


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

    project_to_subspace(
        state=state,
        subspace_basis=sample,
        modes=instruction.modes,
        normalization=normalization,
        connector=state._connector,
    )

    return Result(state=state, samples=samples)


def vacuum(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def _get_normalization(
    probability_map: Mapping[Tuple[int, ...], float], sample: Tuple[int, ...]
) -> float:
    return np.sqrt(1 / probability_map[sample])


def _apply_active_gate_matrix_to_state(
    state_vector: np.ndarray,
    matrix: np.ndarray,
    d: int,
    cutoff: int,
    mode: int,
    connector: BaseConnector,
) -> None:
    @connector.custom_gradient
    def _apply_active_gate_matrix(state_vector, matrix):
        state_vector = connector.preprocess_input_for_custom_gradient(state_vector)
        matrix = connector.preprocess_input_for_custom_gradient(matrix)

        state_index_matrix_list = calculate_state_index_matrix_list(d, cutoff, mode)
        new_state_vector = _calculate_state_vector_after_apply_active_gate(
            state_vector, matrix, state_index_matrix_list, connector
        )
        grad = _create_linear_active_gate_gradient_function(
            state_vector, matrix, state_index_matrix_list, connector
        )
        return new_state_vector, grad

    return _apply_active_gate_matrix(state_vector, matrix)


def _calculate_state_vector_after_apply_active_gate(
    state_vector,
    matrix,
    state_index_matrix_list,
    connector,
):
    np = connector.forward_pass_np

    new_state_vector = np.empty_like(state_vector, dtype=state_vector.dtype)

    is_batch = len(state_vector.shape) == 2

    einsum_string = "ij,jkl->ikl" if is_batch else "ij,jk->ik"

    for state_index_matrix in state_index_matrix_list:
        limit = state_index_matrix.shape[0]
        new_state_vector = connector.assign(
            new_state_vector,
            state_index_matrix,
            np.einsum(
                einsum_string, matrix[:limit, :limit], state_vector[state_index_matrix]
            ),
        )

    return new_state_vector


def _create_linear_active_gate_gradient_function(
    state_vector,
    matrix,
    state_index_matrix_list,
    connector,
):
    def linear_active_gate_gradient(upstream):
        tf = connector._tf
        fallback_np = connector.fallback_np

        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            np = connector.fallback_np
            upstream = upstream.numpy()
        else:
            np = connector.np

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
    space = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    operator = get_creation_operator(
        instruction.modes, space=space, config=state._config
    )

    state.state_vector = operator @ state.state_vector

    return Result(state=state)


def annihilate(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    space = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    operator = get_annihilation_operator(
        instruction.modes, space=space, config=state._config
    )

    state.state_vector = operator @ state.state_vector

    return Result(state=state)


def kerr(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    space = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    xi = instruction._all_params["xi"]
    np = state._np

    mode = instruction.modes[0]

    coefficients = np.exp(1j * xi * np.array([basis[mode] ** 2 for basis in space]))

    # NOTE: Transposition is done here in order to work with batch processing.
    state.state_vector = (coefficients * state.state_vector.T).T

    return Result(state=state)


def cross_kerr(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    space = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    np = state._connector.np

    modes = instruction.modes
    xi = instruction._all_params["xi"]
    for index, basis in enumerate(space):
        coefficient = np.exp(1j * xi * basis[modes[0]] * basis[modes[1]])
        state.state_vector = state._connector.assign(
            state.state_vector, index, state.state_vector[index] * coefficient
        )

    return Result(state=state)


def displacement(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    connector = state._connector

    r = instruction._all_params["r"]
    phi = instruction._all_params["phi"]

    wrapped_get_matrix = connector.decorator(get_single_mode_displacement_operator)

    matrix = wrapped_get_matrix(
        r=r,
        phi=phi,
        cutoff=state._config.cutoff,
        complex_dtype=state._config.complex_dtype,
        connector=connector,
    )

    wrapped_apply = connector.decorator(_apply_active_gate_matrix_to_state)

    state.state_vector = wrapped_apply(
        state.state_vector,
        matrix,
        state.d,
        state._config.cutoff,
        instruction.modes[0],
        connector,
    )

    return Result(state=state)


def _apply_squeezing(state, r, phi, mode):
    connector = state._connector

    wrapped_get_matrix = connector.decorator(get_single_mode_squeezing_operator)

    matrix = wrapped_get_matrix(
        r=r,
        phi=phi,
        cutoff=state._config.cutoff,
        complex_dtype=state._config.complex_dtype,
        connector=connector,
    )

    wrapped_apply = connector.decorator(_apply_active_gate_matrix_to_state)

    state.state_vector = wrapped_apply(
        state.state_vector,
        matrix,
        state.d,
        state._config.cutoff,
        mode,
        connector=connector,
    )


def squeezing(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    _apply_squeezing(
        state,
        r=instruction._all_params["r"],
        phi=instruction._all_params["phi"],
        mode=instruction.modes[0],
    )

    return Result(state=state)


def cubic_phase(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    connector = state._connector

    gamma = instruction._all_params["gamma"]

    wrapped_get_matrix = connector.decorator(get_single_mode_cubic_phase_operator)

    matrix = wrapped_get_matrix(
        gamma=gamma,
        cutoff=state._config.cutoff,
        hbar=state._config.hbar,
        connector=connector,
    )

    wrapped_apply = connector.decorator(_apply_active_gate_matrix_to_state)

    state.state_vector = wrapped_apply(
        state.state_vector,
        matrix,
        state.d,
        state._config.cutoff,
        instruction.modes[0],
        connector,
    )

    return Result(state=state)


def linear(
    state: PureFockState, instruction: gates._ActiveLinearGate, shots: int
) -> Result:
    connector = state._connector
    modes = instruction.modes

    np = connector.np

    passive_block = instruction._get_passive_block(state._connector, state._config)
    active_block = instruction._get_active_block(state._connector, state._config)

    symplectic = connector.block(
        [
            [passive_block, active_block],
            [np.conj(active_block), np.conj(passive_block)],
        ],
    )

    unitary_last, squeezings, unitary_first = euler(symplectic, connector)

    _apply_passive_linear(state, unitary_first, modes, connector)

    for mode, r in zip(instruction.modes, squeezings):
        _apply_squeezing(state, r=r, phi=0.0, mode=mode)

    _apply_passive_linear(state, unitary_last, modes, connector)

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
    occupation_numbers: np.ndarray,
    modes: Optional[Tuple[int, ...]] = None,
) -> None:
    if modes:
        new_occupation_numbers = np.zeros(shape=state.d, dtype=int)

        new_occupation_numbers[modes,] = np.array(occupation_numbers)

        occupation_numbers = new_occupation_numbers

    index = get_index_in_fock_space(occupation_numbers)

    state.state_vector = state._connector.assign(state.state_vector, index, coefficient)


def batch_prepare(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    subprograms = instruction._all_params["subprograms"]
    execute = instruction._all_params["execute"]

    state_vectors = [
        execute(subprogram, shots).state.state_vector for subprogram in subprograms
    ]

    batch_state = BatchPureFockState(
        d=state.d, connector=state._connector, config=state._config
    )

    batch_state._apply_separate_state_vectors(state_vectors)

    return Result(state=batch_state)


def batch_apply(
    state: BatchPureFockState, instruction: Instruction, shots: int
) -> Result:
    subprograms = instruction._all_params["subprograms"]
    execute = instruction._all_params["execute"]

    d = state.d
    connector = state._connector
    config = state._config

    resulting_state_vectors = []

    for state_vector, subprogram in zip(state._batch_state_vectors, subprograms):
        small_state = PureFockState(d=d, connector=connector, config=config)

        small_state.state_vector = state_vector

        resulting_state_vectors.append(
            execute(subprogram, initial_state=small_state).state.state_vector
        )

    state._apply_separate_state_vectors(resulting_state_vectors)

    return Result(state=state)
