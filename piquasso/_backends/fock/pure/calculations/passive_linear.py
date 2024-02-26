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

import numpy as np

from piquasso.api.calculator import BaseCalculator

from ..state import PureFockState
from ...calculations import (
    calculate_interferometer_helper_indices,
    calculate_interferometer_on_fock_space,
    calculate_index_list_for_appling_interferometer,
)
from piquasso.instructions import gates

from piquasso.api.result import Result


def passive_linear(
    state: PureFockState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    calculator = state._calculator

    interferometer: np.ndarray = instruction._get_passive_block(
        state._calculator, state._config
    ).astype(state._config.complex_dtype)

    _apply_passive_linear(state, interferometer, instruction.modes, calculator)

    return Result(state=state)


def _apply_passive_linear(state, interferometer, modes, calculator):
    wrapped = calculator.decorator(_do_apply_passive_linear)

    state.state_vector = wrapped(
        state.state_vector,
        interferometer,
        state.d,
        state._config.cutoff,
        modes,
        calculator,
    )


def _do_apply_passive_linear(
    state_vector, interferometer, d, cutoff, modes, calculator
):
    subspace_transformations = _get_interferometer_on_fock_space(
        interferometer, cutoff, calculator
    )

    return _apply_passive_gate_matrix_to_state(
        state_vector, subspace_transformations, d, cutoff, modes, calculator
    )


def _get_interferometer_on_fock_space(interferometer, cutoff, calculator):
    def _get_interferometer_with_gradient_callback(interferometer):
        interferometer = calculator.preprocess_input_for_custom_gradient(interferometer)
        index_dict = calculate_interferometer_helper_indices(
            d=len(interferometer), cutoff=cutoff
        )

        subspace_representations = calculate_interferometer_on_fock_space(
            interferometer, index_dict, calculator
        )
        grad = _calculate_interferometer_gradient_on_fock_space(
            interferometer,
            calculator,
            subspace_representations,
            index_dict,
        )

        return subspace_representations, grad

    wrapped = calculator.custom_gradient(_get_interferometer_with_gradient_callback)

    return wrapped(interferometer)


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

        subspace_index_tensor = index_dict["subspace_index_tensor"]
        first_subspace_index_tensor = index_dict["first_subspace_index_tensor"]
        first_nonzero_index_tensor = index_dict["first_nonzero_index_tensor"]
        sqrt_first_occupation_numbers_tensor = index_dict[
            "sqrt_first_occupation_numbers_tensor"
        ]
        sqrt_occupation_numbers_tensor = index_dict["sqrt_occupation_numbers_tensor"]

        d = len(interferometer)
        cutoff = len(subspace_index_tensor) + 2

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
                    previous_subspace_grad = subspace_grad[p - 1]

                    first_subspace_indices = fallback_np.asarray(
                        first_subspace_index_tensor[p - 2]
                    )
                    first_nonzero_indices = first_nonzero_index_tensor[p - 2]
                    sqrt_first_occupation_numbers = (
                        sqrt_first_occupation_numbers_tensor[p - 2]
                    )

                    partial_previous_subspace_grad = previous_subspace_grad[
                        first_subspace_indices, :
                    ]

                    sqrt_occupation_numbers = sqrt_occupation_numbers_tensor[p - 2]

                    subspace_indices = subspace_index_tensor[p - 2]

                    first_part_partially_indexed = interferometer[
                        first_nonzero_indices, :
                    ]
                    second_part = partial_previous_subspace_grad[:, subspace_indices]

                    matrix = fallback_np.einsum(
                        "ij,kj,kij->ik",
                        sqrt_occupation_numbers,
                        first_part_partially_indexed,
                        second_part,
                    )

                    matrix = (matrix / sqrt_first_occupation_numbers).T

                    mp1i_indices = fallback_np.where(
                        fallback_np.asarray(first_nonzero_indices) == row_index
                    )[0]

                    occupation_number_sqrts = sqrt_first_occupation_numbers[
                        mp1i_indices
                    ]

                    nm1l_indices = subspace_indices[:, col_index]

                    correction_first_part = subspace_representations[p - 1][
                        first_subspace_indices[mp1i_indices]
                    ][:, nm1l_indices]
                    correction_second_part = (
                        sqrt_occupation_numbers[:, col_index, None]
                        / occupation_number_sqrts
                    )

                    correction_matrix = correction_first_part * correction_second_part.T

                    matrix[mp1i_indices, :] += correction_matrix

                    subspace_grad.append(matrix)

                for i in range(cutoff):
                    full_kl_grad[row_index][col_index] += np.einsum(
                        "ij,ij", upstream[i], fallback_np.conj(subspace_grad[i])
                    )

        return calculator.np.array(full_kl_grad, dtype=interferometer.dtype)

    return interferometer_gradient


def _apply_passive_gate_matrix_to_state(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    d: int,
    cutoff: int,
    modes: Tuple[int, ...],
    calculator: BaseCalculator,
) -> None:
    def _apply_interferometer_matrix(state_vector, subspace_transformations):
        state_vector = calculator.preprocess_input_for_custom_gradient(state_vector)

        subspace_transformations = [
            calculator.preprocess_input_for_custom_gradient(matrix)
            for matrix in subspace_transformations
        ]

        index_list = calculate_index_list_for_appling_interferometer(
            modes,
            d,
            cutoff,
        )

        new_state_vector = _calculate_state_vector_after_interferometer(
            state_vector,
            subspace_transformations,
            index_list,
            calculator,
        )

        grad = _create_linear_passive_gate_gradient_function(
            state_vector,
            subspace_transformations,
            index_list,
            calculator,
        )
        return new_state_vector, grad

    wrapped = calculator.custom_gradient(_apply_interferometer_matrix)

    return wrapped(state_vector, subspace_transformations)


def _calculate_state_vector_after_interferometer(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    index_list: List[np.ndarray],
    calculator: BaseCalculator,
) -> np.ndarray:
    np = calculator.forward_pass_np

    new_state_vector = np.empty_like(state_vector)

    is_batch = len(state_vector.shape) == 2

    einsum_string = "ij,jkl->ikl" if is_batch else "ij,jk->ik"

    for n, indices in enumerate(index_list):
        new_state_vector = calculator.assign(
            new_state_vector,
            indices,
            np.einsum(
                einsum_string, subspace_transformations[n], state_vector[indices]
            ),
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
