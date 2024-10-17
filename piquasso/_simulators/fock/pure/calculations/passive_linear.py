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
import numba as nb

from scipy.special import factorial, comb

from piquasso.api.connector import BaseConnector

from ..state import PureFockState
from ...calculations import (
    calculate_interferometer_helper_indices,
    calculate_index_list_for_appling_interferometer,
)
from piquasso.instructions import gates

from piquasso.api.result import Result


def passive_linear(
    state: PureFockState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    connector = state._connector

    interferometer: np.ndarray = instruction._get_passive_block(
        state._connector, state._config
    ).astype(state._config.complex_dtype)

    _apply_passive_linear(state, interferometer, instruction.modes, connector)

    return Result(state=state)


def _apply_passive_linear(state, interferometer, modes, connector):
    wrapped = connector.decorator(_do_apply_passive_linear)

    state.state_vector = wrapped(
        state.state_vector,
        interferometer,
        state.d,
        state._config.cutoff,
        modes,
        connector,
    )


def _do_apply_passive_linear(state_vector, interferometer, d, cutoff, modes, connector):
    subspace_transformations = _get_interferometer_on_fock_space(
        interferometer, cutoff, connector
    )

    return _apply_passive_gate_matrix_to_state(
        state_vector, subspace_transformations, d, cutoff, modes, connector
    )


def _get_interferometer_on_fock_space(interferometer, cutoff, connector):
    def _get_interferometer_with_gradient_callback(interferometer):
        interferometer = connector.preprocess_input_for_custom_gradient(interferometer)
        index_tuple = calculate_interferometer_helper_indices(
            d=len(interferometer), cutoff=cutoff
        )

        subspace_representations = connector.calculate_interferometer_on_fock_space(
            interferometer, index_tuple
        )
        grad = _calculate_interferometer_gradient_on_fock_space(
            interferometer,
            connector,
            subspace_representations,
            index_tuple,
        )

        return subspace_representations, grad

    wrapped = connector.custom_gradient(_get_interferometer_with_gradient_callback)

    return wrapped(interferometer)


@nb.njit(cache=True)
def _calculate_subspace_grad(
    row_index,
    col_index,
    previous_subspace_representation,
    subspace_index_tuple,
    interferometer,
    previous_subspace_grad,
):
    (
        subspace_indices,
        first_nonzero_indices,
        first_subspace_indices,
        sqrt_occupation_numbers,
        sqrt_first_occupation_numbers,
    ) = subspace_index_tuple

    matrix_dim = sqrt_occupation_numbers.shape[0]

    subspace_grad = np.zeros(shape=(matrix_dim, matrix_dim), dtype=interferometer.dtype)

    for jdx in range(matrix_dim):
        for idx, first_nonzero_index in enumerate(first_nonzero_indices):
            if first_nonzero_index != row_index:
                continue

            subspace_grad[idx, jdx] += (
                previous_subspace_representation[
                    first_subspace_indices[idx],
                    subspace_indices[jdx, col_index],
                ]
                * sqrt_occupation_numbers[jdx, col_index]
            )

        for idx in range(matrix_dim):
            for kdx in range(sqrt_occupation_numbers.shape[1]):
                subspace_grad[idx, jdx] += (
                    sqrt_occupation_numbers[jdx, kdx]
                    * interferometer[first_nonzero_indices[idx], kdx]
                    * previous_subspace_grad[
                        first_subspace_indices[idx], subspace_indices[jdx, kdx]
                    ]
                )

            subspace_grad[idx, jdx] /= sqrt_first_occupation_numbers[idx]

    return subspace_grad


def _calculate_interferometer_gradient_on_fock_space(
    interferometer, connector, subspace_representations, index_tuple
):
    def interferometer_gradient(*upstream):
        tf = connector._tf
        fallback_np = connector.fallback_np

        static_valued = tf.get_static_value(upstream[0]) is not None

        if static_valued:
            np = connector.fallback_np
            upstream = [x.numpy() for x in upstream]
        else:
            np = connector.np

        d = len(interferometer)
        cutoff = len(index_tuple[0]) + 2

        (
            subspace_index_tensor,
            first_nonzero_index_tensor,
            first_subspace_index_tensor,
            sqrt_occupation_numbers_tensor,
            sqrt_first_occupation_numbers_tensor,
        ) = index_tuple

        full_kl_grad = []
        for row_index in range(d):
            full_kl_grad.append([0] * d)
            for col_index in range(d):
                second_subspace_grad = fallback_np.zeros(
                    shape=interferometer.shape, dtype=interferometer.dtype
                )
                second_subspace_grad[row_index, col_index] = 1.0

                previous_subspace_grad = second_subspace_grad

                for p in range(2, cutoff):
                    previous_subspace_representation = subspace_representations[p - 1]

                    subspace_index_tuple = (
                        subspace_index_tensor[p - 2],
                        first_nonzero_index_tensor[p - 2],
                        first_subspace_index_tensor[p - 2],
                        sqrt_occupation_numbers_tensor[p - 2],
                        sqrt_first_occupation_numbers_tensor[p - 2],
                    )

                    subspace_grad = _calculate_subspace_grad(
                        row_index,
                        col_index,
                        previous_subspace_representation,
                        subspace_index_tuple,
                        interferometer,
                        previous_subspace_grad,
                    )

                    full_kl_grad[row_index][col_index] += np.einsum(
                        "ij,ij", upstream[p], fallback_np.conj(subspace_grad)
                    )

                    previous_subspace_grad = subspace_grad

        partial_result = connector.np.array(full_kl_grad, dtype=interferometer.dtype)
        one_particle_term = upstream[1]
        result = partial_result + one_particle_term

        return result

    return interferometer_gradient


def _apply_passive_gate_matrix_to_state(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    d: int,
    cutoff: int,
    modes: Tuple[int, ...],
    connector: BaseConnector,
) -> None:
    def _apply_interferometer_matrix(state_vector, subspace_transformations):
        state_vector = connector.preprocess_input_for_custom_gradient(state_vector)

        subspace_transformations = [
            connector.preprocess_input_for_custom_gradient(matrix)
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
            connector,
        )

        grad = _create_linear_passive_gate_gradient_function(
            state_vector,
            subspace_transformations,
            index_list,
            connector,
        )
        return new_state_vector, grad

    wrapped = connector.custom_gradient(_apply_interferometer_matrix)

    return wrapped(state_vector, subspace_transformations)


def _calculate_state_vector_after_interferometer(
    state_vector: np.ndarray,
    subspace_transformations: List[np.ndarray],
    index_list: List[np.ndarray],
    connector: BaseConnector,
) -> np.ndarray:
    np = connector.forward_pass_np

    new_state_vector = np.empty_like(state_vector)

    is_batch = len(state_vector.shape) == 2

    einsum_string = "ij,jkl->ikl" if is_batch else "ij,jk->ik"

    for n, indices in enumerate(index_list):
        new_state_vector = connector.assign(
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
    connector: BaseConnector,
) -> Callable:
    def applying_interferometer_gradient(upstream):
        tf = connector._tf
        fallback_np = connector.fallback_np

        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            np = connector.fallback_np
            upstream = upstream.numpy()
        else:
            np = connector.np

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


@lru_cache(maxsize=None)
def _beamsplitter5050_coeff(n, m, N):
    return (
        2 ** (-(n + m) / 2)
        * np.sqrt(factorial(N) * factorial(n + m - N) / (factorial(n) * factorial(m)))
        * (-1) ** n
        * np.sum([comb(n, j) * comb(m, N - j) * (-1) ** j for j in range(N + 1)])
    )


@lru_cache(maxsize=None)
def _calculate_beamsplitter5050_for_symmetric_subspace(number_of_particles):
    """
    NOTE: Further optimizations may be done here.
    """
    dimension = number_of_particles + 1

    matrix = np.empty(shape=(dimension, dimension))

    for row_index, n in enumerate(range(number_of_particles, -1, -1)):
        m = number_of_particles - n
        for col_index, N in enumerate(range(number_of_particles, -1, -1)):
            matrix[row_index, col_index] = _beamsplitter5050_coeff(n, m, N)

    return matrix


@lru_cache(maxsize=None)
def _calculate_beamsplitter5050_on_fock_space(cutoff):
    return [
        _calculate_beamsplitter5050_for_symmetric_subspace(number_of_particles)
        for number_of_particles in range(cutoff)
    ]


def beamsplitter5050(
    state: PureFockState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    _apply_beamsplitter5050(state, instruction.modes)

    return Result(state=state)


def _apply_beamsplitter5050(state, modes):
    cutoff = state._config.cutoff
    subspace_transformations = _calculate_beamsplitter5050_on_fock_space(cutoff)

    state.state_vector = _apply_passive_gate_matrix_to_state(
        state_vector=state.state_vector,
        subspace_transformations=subspace_transformations,
        d=state.d,
        cutoff=cutoff,
        modes=modes,
        connector=state._connector,
    )
