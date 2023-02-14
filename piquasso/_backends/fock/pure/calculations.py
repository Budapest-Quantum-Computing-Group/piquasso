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
    def _get_interferometer_with_gradient_callback(interferometer):
        interferometer = calculator.maybe_convert_to_numpy(interferometer)
        index_dict = _calculate_interferometer_helper_indices(space)

        subspace_representations = _calculate_interferometer_on_fock_space(
            interferometer, index_dict
        )
        grad = _calculate_interferometer_gradient_on_fock_space(
            interferometer, calculator, subspace_representations, index_dict
        )

        return subspace_representations, grad

    wrapped = calculator.custom_gradient(_get_interferometer_with_gradient_callback)

    return wrapped(interferometer)


def _calculate_interferometer_helper_indices(space):
    d = space.d
    cutoff = space.cutoff
    space = [np.array(element, dtype=int) for element in space]
    indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

    subspace_index_tensor = []
    first_subspace_index_tensor = []

    nonzero_index_tensor = []
    first_nonzero_index_tensor = []

    sqrt_occupation_numbers_tensor = []
    first_occupation_numbers_tensor = []

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]
        subspace = space[indices[n - 1] : indices[n]]

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

        subspace_index_tensor.append(subspace_indices)
        first_nonzero_index_tensor.append(first_nonzero_indices)

        nonzero_index_tensor.append(nonzero_indices)
        first_subspace_index_tensor.append(first_subspace_indices)

        sqrt_occupation_numbers_tensor.append(sqrt_occupation_numbers)
        first_occupation_numbers_tensor.append(first_occupation_numbers)

    return {
        "subspace_index_tensor": subspace_index_tensor,
        "first_nonzero_index_tensor": first_nonzero_index_tensor,
        "nonzero_index_tensor": nonzero_index_tensor,
        "first_subspace_index_tensor": first_subspace_index_tensor,
        "sqrt_occupation_numbers_tensor": sqrt_occupation_numbers_tensor,
        "first_occupation_numbers_tensor": first_occupation_numbers_tensor,
        "indices": indices,
    }


def _calculate_interferometer_on_fock_space(interferometer, index_dict):

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

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)

    indices = index_dict["indices"]
    cutoff = len(indices)

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]

        subspace_indices = index_dict["subspace_index_tensor"][n - 2]
        first_subspace_indices = index_dict["first_subspace_index_tensor"][n - 2]

        nonzero_indices = index_dict["nonzero_index_tensor"][n - 2]
        first_nonzero_indices = index_dict["first_nonzero_index_tensor"][n - 2]

        sqrt_occupation_numbers = index_dict["sqrt_occupation_numbers_tensor"][n - 2]
        first_occupation_numbers = index_dict["first_occupation_numbers_tensor"][n - 2]

        matrix = []

        for index in range(size):
            first_part = (
                sqrt_occupation_numbers[index]
                * interferometer[np.ix_(first_nonzero_indices, nonzero_indices[index])]
            )
            second_part = subspace_representations[n - 1][
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


def _calculate_interferometer_gradient_on_fock_space(
    interferometer, calculator, subspace_representations, index_dict
):
    def interferometer_gradient(*upstream):
        import time
        time_sum = 0.0
        tf = calculator._tf

        indices = index_dict["indices"]
        subspace_index_tensor = index_dict["subspace_index_tensor"]
        first_subspace_index_tensor = index_dict["first_subspace_index_tensor"]
        nonzero_index_tensor = index_dict["nonzero_index_tensor"]
        first_nonzero_index_tensor = index_dict["first_nonzero_index_tensor"]
        sqrt_occupation_numbers_tensor = index_dict["sqrt_occupation_numbers_tensor"]
        first_occupation_numbers_tensor = index_dict["first_occupation_numbers_tensor"]

        d = len(interferometer)
        cutoff = len(indices)

        full_kl_grad = []
        for row_index in range(d):
            full_kl_grad.append([0] * d)
            for col_index in range(d):
                subspace_grad = []
                subspace_grad.append(np.array([[0]], dtype=complex))
                second_subspace = np.zeros(shape=interferometer.shape, dtype=complex)
                second_subspace[row_index, col_index] = 1
                subspace_grad.append(second_subspace)

                for p in range(2, cutoff):
                    size = indices[p] - indices[p - 1]
                    previous_subspace_grad = subspace_grad[p - 1]
                    matrix = np.zeros(shape=(size, size), dtype=complex)

                    subspace_indices = subspace_index_tensor[p - 2]
                    first_subspace_indices = np.asarray(
                        first_subspace_index_tensor[p - 2]
                    )
                    nonzero_indices = nonzero_index_tensor[p - 2]
                    first_nonzero_indices = first_nonzero_index_tensor[p - 2]
                    sqrt_occupation_numbers = sqrt_occupation_numbers_tensor[p - 2]
                    first_occupation_numbers = first_occupation_numbers_tensor[p - 2]

                    start_time = time.time()
                    for n_index in range(size):
                        first_part = (
                            sqrt_occupation_numbers[n_index]
                            * interferometer[
                                np.ix_(first_nonzero_indices, nonzero_indices[n_index])
                            ]
                        )
                        second_part = previous_subspace_grad[
                            np.ix_(first_subspace_indices, subspace_indices[n_index])
                        ]
                        full = np.einsum("ij,ij->i", first_part, second_part)
                        matrix[:, n_index] = full / np.sqrt(first_occupation_numbers)

                    time_sum += time.time() - start_time

                    mp1i_indices = np.where(
                        np.asarray(first_nonzero_indices) == row_index
                    )[0]
                    col_nonzero_indices_index = []
                    col_nonzero_indices = []
                    for index in range(size):
                        if col_index in nonzero_indices[index]:
                            col_nonzero_indices_index.append(
                                nonzero_indices[index].tolist().index(col_index)
                            )
                            col_nonzero_indices.append(index)

                    for index in range(len(col_nonzero_indices_index)):
                        nm1l_index = subspace_indices[col_nonzero_indices[index]][
                            col_nonzero_indices_index[index]
                        ]
                        matrix[mp1i_indices, col_nonzero_indices[index]] += (
                            sqrt_occupation_numbers[col_nonzero_indices[index]][
                                col_nonzero_indices_index[index]
                            ]
                            / np.sqrt(first_occupation_numbers[mp1i_indices])
                            * subspace_representations[p - 1][
                                first_subspace_indices[mp1i_indices], nm1l_index
                            ]
                        )


                    subspace_grad.append(matrix)

                for i in range(cutoff):
                    full_kl_grad[row_index][col_index] += tf.einsum(
                        "ij,ij", upstream[i], np.conj(subspace_grad[i])
                    )

        print("interferometer_gradient TIME:", time_sum)
        return calculator.np.array(full_kl_grad)

    return interferometer_gradient


def passive_linear(
    state: PureFockState, instruction: Instruction, shots: int
) -> Result:
    calculator = state._calculator

    interferometer: np.ndarray = instruction._all_params["passive_block"].astype(np.complex128)

    subspace = FockSpace(
        d=len(interferometer), cutoff=state._space.cutoff, calculator=calculator
    )

    subspace_transformations = list(
        _get_interferometer_on_fock_space(interferometer, subspace, calculator)
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
    calculator = state._calculator
    state_vector = state._state_vector
    space = state._space

    def _apply_interferometer_matrix(state_vector, subspace_transformations):
        state_vector = calculator.maybe_convert_to_numpy(state_vector)

        subspace_transformations = [calculator.maybe_convert_to_numpy(matrix) for matrix in subspace_transformations]

        index_list = _calculate_index_list_for_appling_interferometer(
            modes, space, calculator,
        )

        new_state_vector = _calculate_state_vector_after_interferometer(
            state_vector, subspace_transformations, index_list,
        )

        grad = _create_linear_passive_gate_gradient_function(
            state_vector, subspace_transformations, index_list, calculator,
        )
        return new_state_vector, grad

    wrapped = calculator.custom_gradient(_apply_interferometer_matrix)

    state._state_vector = wrapped(state_vector, subspace_transformations)

def _calculate_index_list_for_appling_interferometer(
    modes: Tuple[int, ...],
    space: FockSpace,
    calculator,
) -> List[np.ndarray]:
    cutoff = space.cutoff
    d = space.d

    subspace = FockSpace(
        d=len(modes), cutoff=space.cutoff, calculator=calculator
    )
    auxiliary_subspace = FockSpace(
        d=d - len(modes), cutoff=space.cutoff, calculator=calculator
    )

    indices = [
        cutoff_cardinality(cutoff=n, d=len(modes)) for n in range(cutoff + 1)
    ]
    auxiliary_indices = [
        cutoff_cardinality(cutoff=n, d=d-len(modes)) for n in range(cutoff + 1)
    ]

    auxiliary_modes = get_auxiliary_modes(d, modes)
    all_occupation_numbers = np.zeros(d, dtype=int)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff-n]
        state_index_matrix = np.empty(shape=(size, auxiliary_size), dtype=int)
        for idx1, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[: auxiliary_size]
        ):
            all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers
            for idx2, column_vector_on_subspace in enumerate(n_particle_subspace):
                all_occupation_numbers[(modes,)] = column_vector_on_subspace
                column_index = get_index_in_fock_space(tuple(all_occupation_numbers))
                state_index_matrix[idx2, idx1] = column_index

        index_list.append(state_index_matrix)

    return index_list


def _calculate_state_vector_after_interferometer(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    index_list: List[np.ndarray]
) -> None:
    new_state_vector = np.empty_like(state_vector)

    for n, indices in enumerate(index_list):
        new_state_vector[indices] = subspace_transformations[n] @ state_vector[indices]

    return new_state_vector


def _create_linear_passive_gate_gradient_function(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    index_list: List[np.ndarray],
    calculator,
):
    def applying_interferometer_gradient(upstream):
        unordered_gradient_by_initial_state = []
        order_by = []

        gradient_by_matrix = []

        for n, indices in enumerate(index_list):
            matrix = subspace_transformations[n]
            sliced_upstream = upstream[indices]

            order_by.append(indices.reshape(-1))
            product = matrix @ sliced_upstream
            unordered_gradient_by_initial_state.append(product.reshape(-1))

            state_vector_slice = state_vector[indices]
            gradient_by_matrix.append(
                calculator._tf.einsum("ij,kj->ki", state_vector_slice, sliced_upstream)
            )

        gradient_by_initial_state = calculator.np.concatenate(
            unordered_gradient_by_initial_state
        )[np.concatenate(order_by).argsort()]

        return gradient_by_initial_state, gradient_by_matrix

    return applying_interferometer_gradient


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
    new_state_vector = np.empty_like(state_vector, dtype=np.csingle)

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

        unordered_gradient_by_initial_state = []
        order_by = []

        gradient_by_matrix = tf.zeros(shape=(cutoff, cutoff), dtype=np.complex128)

        for indices in state_index_matrix_list:
            limit = indices.shape[0]

            matrix_slice = matrix[:limit, :limit].T

            order_by.append(indices.reshape(-1))
            unordered_gradient_by_initial_state.append(
                (matrix_slice @ upstream[indices]).reshape(-1)
            )

            gradient_by_matrix += tf.pad(
                tf.einsum("ij,kj->ki", state_vector[indices], upstream[indices]),
                [[0, cutoff-limit], [0, cutoff-limit]],
                "CONSTANT",
            )

        gradient_by_initial_state = calculator.np.concatenate(
            unordered_gradient_by_initial_state
        )[np.concatenate(order_by).argsort()]

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
    xi_vector = instruction._all_params["xi_vector"]
    np = state._np

    for mode_index, mode in enumerate(instruction.modes):
        xi = xi_vector[mode_index]

        coefficients = np.exp(
            1j * xi * np.array([basis[mode] ** 2 for basis in state._space])
        )

        state._state_vector = coefficients * state._state_vector

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

    amplitudes = np.abs(instruction._all_params["displacement_vector"]).astype(np.float32)
    angles = np.angle(instruction._all_params["displacement_vector"]).astype(np.float32)

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_displacement_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        _apply_active_gate_matrix_to_state(state, matrix, mode)

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

        _apply_active_gate_matrix_to_state(state, matrix, mode)

    state.normalize()

    return Result(state=state)


def cubic_phase(state: PureFockState, instruction: Instruction, shots: int) -> Result:
    gamma_vector = instruction._all_params["gamma_vector"]
    hbar = state._config.hbar

    for index, mode in enumerate(instruction.modes):
        matrix = state._space.get_single_mode_cubic_phase_operator(
            gamma=gamma_vector[index], hbar=hbar, calculator=state._calculator
        )
        _apply_active_gate_matrix_to_state(state, matrix, mode)

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
