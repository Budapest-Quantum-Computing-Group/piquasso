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

from typing import Tuple, List, Callable

from functools import lru_cache

import numpy as np

from piquasso._math.fock import FockSpace, cutoff_cardinality

from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_subspace,
    get_auxiliary_modes,
)
from piquasso.api.calculator import BaseCalculator

from ..state import PureFockState

from piquasso.instructions import gates

from piquasso.api.result import Result


def passive_linear(
    state: PureFockState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    calculator = state._calculator

    interferometer: np.ndarray = instruction._get_passive_block(
        state._calculator, state._config
    ).astype(np.complex128)

    subspace = state._get_subspace(dim=len(interferometer))

    subspace_transformations = _get_interferometer_on_fock_space(
        interferometer, subspace, calculator
    )

    _apply_passive_gate_matrix_to_state(
        state, subspace_transformations, instruction.modes
    )

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


@lru_cache(maxsize=None)
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

        first_part_partially_indexed = interferometer[first_nonzero_indices, :]
        second_part_partially_indexed = subspace_representations[n - 1][
            first_subspace_indices, :
        ]

        matrix = []

        for index in range(size):
            first_part = (
                sqrt_occupation_numbers[index]
                * first_part_partially_indexed[:, nonzero_indices[index]]
            )
            second_part = second_part_partially_indexed[:, subspace_indices[index]]
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
        tf = calculator._tf
        fallback_np = calculator.fallback_np

        static_valued = tf.get_static_value(upstream[0]) is not None

        if static_valued:
            np = calculator.fallback_np
            upstream = [x.numpy() for x in upstream]
        else:
            np = calculator.np

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
                subspace_grad.append(
                    fallback_np.array([[0]], dtype=interferometer.dtype)
                )
                second_subspace = fallback_np.zeros(
                    shape=interferometer.shape, dtype=interferometer.dtype
                )
                second_subspace[row_index, col_index] = 1
                subspace_grad.append(second_subspace)

                for p in range(2, cutoff):
                    size = indices[p] - indices[p - 1]
                    previous_subspace_grad = subspace_grad[p - 1]
                    matrix = fallback_np.zeros(
                        shape=(size, size), dtype=interferometer.dtype
                    )

                    subspace_indices = subspace_index_tensor[p - 2]
                    first_subspace_indices = fallback_np.asarray(
                        first_subspace_index_tensor[p - 2]
                    )
                    nonzero_indices = nonzero_index_tensor[p - 2]
                    first_nonzero_indices = first_nonzero_index_tensor[p - 2]
                    sqrt_occupation_numbers = sqrt_occupation_numbers_tensor[p - 2]
                    first_occupation_numbers = first_occupation_numbers_tensor[p - 2]

                    partial_interferometer = interferometer[first_nonzero_indices, :]
                    partial_previous_subspace_grad = previous_subspace_grad[
                        first_subspace_indices, :
                    ]

                    for n_index in range(size):
                        first_part = (
                            sqrt_occupation_numbers[n_index]
                            * partial_interferometer[:, nonzero_indices[n_index]]
                        )
                        second_part = partial_previous_subspace_grad[
                            :, subspace_indices[n_index]
                        ]
                        full = fallback_np.einsum("ij,ij->i", first_part, second_part)
                        matrix[:, n_index] = full

                    matrix = (matrix.T / fallback_np.sqrt(first_occupation_numbers)).T

                    mp1i_indices = fallback_np.where(
                        fallback_np.asarray(first_nonzero_indices) == row_index
                    )[0]

                    occupation_number_sqrts = fallback_np.sqrt(
                        first_occupation_numbers[mp1i_indices]
                    )

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
                            / occupation_number_sqrts
                            * subspace_representations[p - 1][
                                first_subspace_indices[mp1i_indices], nm1l_index
                            ]
                        )

                    subspace_grad.append(matrix)

                for i in range(cutoff):
                    full_kl_grad[row_index][col_index] += np.einsum(
                        "ij,ij", upstream[i], fallback_np.conj(subspace_grad[i])
                    )

        return calculator.np.array(full_kl_grad)

    return interferometer_gradient


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

        subspace_transformations = [
            calculator.maybe_convert_to_numpy(matrix)
            for matrix in subspace_transformations
        ]

        index_list = _calculate_index_list_for_appling_interferometer(
            modes,
            space,
        )

        new_state_vector = _calculate_state_vector_after_interferometer(
            state_vector,
            subspace_transformations,
            index_list,
        )

        grad = _create_linear_passive_gate_gradient_function(
            state_vector,
            subspace_transformations,
            index_list,
            calculator,
        )
        return new_state_vector, grad

    wrapped = calculator.custom_gradient(_apply_interferometer_matrix)

    state._state_vector = wrapped(state_vector, subspace_transformations)


@lru_cache(maxsize=None)
def _calculate_index_list_for_appling_interferometer(
    modes: Tuple[int, ...],
    space: FockSpace,
) -> List[np.ndarray]:
    cutoff = space.cutoff
    d = space.d
    calculator = space._calculator

    subspace = FockSpace(
        d=len(modes), cutoff=space.cutoff, calculator=calculator, config=space.config
    )
    auxiliary_subspace = FockSpace(
        d=d - len(modes),
        cutoff=space.cutoff,
        calculator=calculator,
        config=space.config,
    )

    indices = [cutoff_cardinality(cutoff=n, d=len(modes)) for n in range(cutoff + 1)]
    auxiliary_indices = [
        cutoff_cardinality(cutoff=n, d=d - len(modes)) for n in range(cutoff + 1)
    ]

    auxiliary_modes = get_auxiliary_modes(d, modes)
    all_occupation_numbers = np.zeros(d, dtype=int)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff - n]
        state_index_matrix = np.empty(shape=(size, auxiliary_size), dtype=int)
        for idx1, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[:auxiliary_size]
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
    index_list: List[np.ndarray],
) -> np.ndarray:
    new_state_vector = np.empty_like(state_vector)

    is_batch = len(state_vector.shape) == 2

    einsum_string = "ij,jkl->ikl" if is_batch else "ij,jk->ik"

    for n, indices in enumerate(index_list):
        new_state_vector[indices] = np.einsum(
            einsum_string, subspace_transformations[n], state_vector[indices]
        )

    return new_state_vector


def _create_linear_passive_gate_gradient_function(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    index_list: List[np.ndarray],
    calculator: BaseCalculator,
) -> Callable:
    def applying_interferometer_gradient(upstream):
        tf = calculator._tf
        fallback_np = calculator.fallback_np

        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            np = calculator.fallback_np
            upstream = upstream.numpy()
        else:
            np = calculator.np

        is_batch = len(state_vector.shape) == 2

        matrix_einsum_string = "ijl,kjl->ki" if is_batch else "ij,kj->ki"
        initial_state_einsum_string = "ji,jkl->ikl" if is_batch else "ji,jk->ik"

        reshape_arg = (-1, state_vector.shape[1]) if is_batch else (-1,)

        unordered_gradient_by_initial_state = []
        order_by = []

        gradient_by_matrix = []

        conjugated_state_vector = np.conj(state_vector)

        for n, indices in enumerate(index_list):
            matrix = np.conj(subspace_transformations[n])
            sliced_upstream = upstream[indices]
            state_vector_slice = conjugated_state_vector[indices]

            order_by.append(indices.reshape(-1))
            product = np.einsum(initial_state_einsum_string, matrix, sliced_upstream)
            unordered_gradient_by_initial_state.append(product.reshape(*reshape_arg))

            gradient_by_matrix.append(
                np.einsum(matrix_einsum_string, state_vector_slice, sliced_upstream)
            )

        gradient_by_initial_state = np.concatenate(unordered_gradient_by_initial_state)[
            fallback_np.concatenate(order_by).argsort()
        ]

        if static_valued:
            gradient_by_initial_state = tf.constant(gradient_by_initial_state)
            gradient_by_matrix = [tf.constant(matrix) for matrix in gradient_by_matrix]

        return gradient_by_initial_state, gradient_by_matrix

    return applying_interferometer_gradient
