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

from typing import Optional, Tuple, Mapping, List

import random
import numpy as np

from piquasso._math.fock import FockSpace, cutoff_cardinality

from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_subspace,
    get_auxiliary_modes,
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

    @calculator.custom_gradient
    def f(interferometer):

        interferometer = calculator.maybe_convert_to_numpy(interferometer)

        subspace_representations = _calculate_interferometer_on_fock_space(
            interferometer, space
        )
        grad = _calculate_interferometer_gradient_on_fock_space(
            interferometer, space, calculator, subspace_representations
        )

        return subspace_representations, grad

    return f(interferometer)

def _calculate_interferometer_on_fock_space(interferometer, space):

    """Calculates finite representation of interferometer in the Fock space.
    The function assumes the knowledge of the 1-particle unitary.

    Sources:
    - Fast optimization of parametrized quantum optical circuits
    (https://quantum-journal.org/papers/q-2020-11-30-366/)

    Args:
        interferometer (numpy.ndarray): The 1-particle unitary
        space (FockSpace): List of basis elements on the Fock space.

    Returns:
        numpy.ndarray: Finite representation of interferometer in the Fock space
    """

    cutoff = space.cutoff
    d = space.d

    space = [np.array(element, dtype=int) for element in space]

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)

    indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]

        previous_representation = subspace_representations[n - 1]

        subspace = space[indices[n - 1] : indices[n]]

        matrix = []

        subspace_indices = []
        first_subspace_indices = []

        nonzero_indices = []
        first_nonzero_indices = []

        sqrt_occupation_numbers = []
        first_occupation_numbers = np.empty(size)

        for index, vector in enumerate(subspace):
            nonzero_multiindex = np.nonzero(vector)[0]
            first_nonzero_multiindex = nonzero_multiindex[0]

            subspace_multiindex = []
            for nonzero_index in nonzero_multiindex:
                vector[nonzero_index] -= 1
                subspace_multiindex.append(get_index_in_fock_subspace(tuple(vector)))
                vector[nonzero_index] += 1

            subspace_indices.append(subspace_multiindex)
            first_subspace_indices.append(subspace_multiindex[0])

            nonzero_indices.append(nonzero_multiindex)
            first_nonzero_indices.append(first_nonzero_multiindex)

            sqrt_occupation_numbers.append(np.sqrt(vector[nonzero_multiindex]))
            first_occupation_numbers[index] = vector[first_nonzero_multiindex]

        for index in range(size):
            first_part = (
                sqrt_occupation_numbers[index]
                * interferometer[np.ix_(first_nonzero_indices, nonzero_indices[index])]
            )
            second_part = previous_representation[
                np.ix_(first_subspace_indices, subspace_indices[index])
            ]
            matrix.append(np.einsum("ij,ij->i", first_part, second_part))

        new_subspace_representation = np.transpose(
            np.array(matrix) / np.sqrt(first_occupation_numbers)
        )

        subspace_representations.append(
            new_subspace_representation.astype(interferometer.dtype)
        )

    return subspace_representations

def _calculate_interferometer_gradient_on_fock_space(interferometer, full_space, calculator, subspace_representations):

    def grad(*upstream):
        cutoff = full_space.cutoff
        d = full_space.d
        tf = calculator._tf
        space = [np.array(element, dtype=int) for element in full_space]

        full_kl_grad = [] # d_kl e
        for k in range(d):
            full_kl_grad.append([0]*d)
            for l in range(d):
                subspace_grad = []  # d_kl U
                subspace_grad.append(np.array([[0]], dtype=complex)) # d_kl U_0,0
                second_subspace = np.zeros(shape=interferometer.shape, dtype=complex)
                second_subspace[k, l] = 1  # d_kl U_1,1
                subspace_grad.append(second_subspace)

                indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

                for p in range(2, cutoff):

                    size = indices[p] - indices[p - 1]
                    previous_subspace_grad = subspace_grad[p - 1]
                    subspace = space[indices[p - 1] : indices[p]]
                    matrix = np.zeros(shape=(size, size), dtype=complex)  # V^(symm tensor n)

                    for mp1i in subspace:
                        i = np.nonzero(mp1i)[0][0]
                        m = np.copy(mp1i)
                        m[i] -= 1
                        mp1i_index = get_index_in_fock_subspace(tuple(mp1i))
                        for n in subspace:
                            n_index = get_index_in_fock_subspace(tuple(n))
                            if k == i and n[l] != 0:
                                nm1l = np.copy(n)
                                nm1l[l] -= 1
                                matrix[mp1i_index, n_index] += (
                                    np.sqrt(n[l] / (m[i] + 1)) * subspace_representations[p - 1][get_index_in_fock_subspace(tuple(m)), get_index_in_fock_subspace(tuple(nm1l))]
                                )

                            for j in range(d):
                                if n[j] != 0:
                                    nm1j = np.copy(n)
                                    nm1j[j] -= 1
                                    if mp1i_index == 0 and n_index == 1:
                                        pass
                                    gyokos = np.sqrt(n[j] / (m[i] + 1))
                                    inter = interferometer[i, j]
                                    prev = previous_subspace_grad[get_index_in_fock_subspace(tuple(m)), get_index_in_fock_subspace(tuple(nm1j))]
                                    full = gyokos * inter * prev
                                    matrix[mp1i_index, n_index] += full

                    subspace_grad.append(matrix)
                breakpoint()
                for i in range(len(subspace_grad)):
                    full_kl_grad[k][l] += tf.einsum("ij,ij", upstream[i], np.conj(subspace_grad[i]))

        return calculator.np.array(full_kl_grad)

    return grad


def passive_linear(
    state: PureFockState, instruction: Instruction, shots: int
) -> Result:
    calculator = state._calculator

    interferometer: np.ndarray = instruction._all_params["passive_block"]

    subspace = FockSpace(
        d=len(interferometer), cutoff=state._space.cutoff, calculator=calculator
    )

    subspace_transformations = list(_get_interferometer_on_fock_space(
        interferometer, subspace, calculator)
    )

    _apply_passive_gate_matrix_to_state(
        state, subspace_transformations, instruction.modes
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


def _apply_passive_gate_matrix_to_state(
    state: PureFockState,
    subspace_transformations: List[np.ndarray],
    modes: Tuple[int, ...],
) -> None:
    np = state._np
    fallback_np = state._calculator.fallback_np

    space = state._space
    cutoff = space.cutoff
    subspace = FockSpace(
        d=len(modes), cutoff=space.cutoff, calculator=state._calculator
    )
    subspace_indices = [
        cutoff_cardinality(cutoff=n, d=len(modes)) for n in range(cutoff + 1)
    ]

    d = state._space.d
    auxiliary_subspace = FockSpace(
        d=d - len(modes), cutoff=space.cutoff, calculator=state._calculator
    )

    new_state_vector = [0.0] * len(state._state_vector)

    auxiliary_modes = state._get_auxiliary_modes(modes)
    all_occupation_numbers = fallback_np.zeros(d, dtype=int)

    for auxiliary_occupation_numbers in auxiliary_subspace:
        all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers

        for n in range(cutoff - sum(auxiliary_occupation_numbers)):
            state_indices = []

            matrix = subspace_transformations[n]

            for column_vector_on_subspace in subspace[
                subspace_indices[n] : subspace_indices[n + 1]
            ]:
                all_occupation_numbers[(modes,)] = column_vector_on_subspace
                column_index = get_index_in_fock_space(tuple(all_occupation_numbers))
                state_indices.append(column_index)

            for matrix_index, state_index in enumerate(state_indices):
                new_state_vector[state_index] = np.dot(
                    matrix[matrix_index], state._state_vector[state_indices]
                )

    state._state_vector = np.stack(new_state_vector)


def _apply_active_gate_matrix_to_state(
    state: PureFockState,
    matrix: np.ndarray,
    mode: int,
) -> None:
    calculator = state._calculator
    state_vector = state._state_vector
    space = state._space
    auxiliary_subspace = state._auxiliary_subspace

    @calculator.custom_gradient
    def f(state_vector, matrix):
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

    state._state_vector = f(state_vector, matrix)


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
    new_state_vector = np.empty_like(state_vector, dtype=complex)

    for state_index_matrix in state_index_matrix_list:
        limit = state_index_matrix.shape[0]
        new_state_vector[state_index_matrix] = (
            matrix[:limit, :limit] @ state_vector[state_index_matrix]
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
        cutoff = len(state_index_matrix_list)

        sort_by = np.array([], dtype=int)
        accumulator = []

        for index_matrix in state_index_matrix_list:
            limit = index_matrix.shape[0]
            res = matrix[:limit, :limit].T @ upstream[index_matrix]

            sort_by = np.concatenate([sort_by, index_matrix.reshape(-1)])
            accumulator.append(res.reshape(-1))

        result_initial_state_grad = calculator.np.concatenate(accumulator)[
            sort_by.argsort()
        ]

        # NOTE: Very ugly solution, but tensorflow tensors could not be assigned
        # elementwise, so one has to do calculate it this way.
        result_matrix_grad_as_list = []
        for _ in range(cutoff):
            result_matrix_grad_as_list.append([0.0] * cutoff)

        for index_matrix in state_index_matrix_list:
            limit = index_matrix.shape[0]
            partial_result = tf.einsum(
                "ik,jk->ij",
                upstream[index_matrix[:limit, :]],
                state_vector[index_matrix[:limit, :]],
            )
            for i in range(limit):
                for j in range(limit):
                    result_matrix_grad_as_list[i][j] += partial_result[i, j]

        result_matrix_grad = calculator.np.array(result_matrix_grad_as_list)

        return result_initial_state_grad, result_matrix_grad

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

        _apply_active_gate_matrix_to_state(
            state,
            matrix,
            mode,
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

        _apply_active_gate_matrix_to_state(
            state,
            matrix,
            mode,
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
        _apply_active_gate_matrix_to_state(
            state,
            matrix,
            mode,
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