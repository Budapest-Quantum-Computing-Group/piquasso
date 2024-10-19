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

import numpy as np
from scipy.special import factorial

from functools import partial


from piquasso._math.combinatorics import partitions

"""
This is a contribution from `theboss`, see https://github.com/Tomev-CTP/theboss.

The original code has been re-implemented and adapted to Piquasso.
"""

__author__ = "Tomasz Rybotycki"


def calculate_state_vector(interferometer, initial_state, config, connector):
    """
    Calculates the state vector on the particle subspace defined by `initial_state`.
    """

    np = connector.np
    fallback_np = connector.fallback_np

    possible_outputs = partitions(
        particles=connector.fallback_np.sum(initial_state),
        boxes=len(initial_state),
    )

    state_vector = np.empty(possible_outputs.shape[0], dtype=config.complex_dtype)

    input = initial_state

    for idx, output in enumerate(possible_outputs):
        permanent = connector.permanent(interferometer, cols=input, rows=output)

        state_vector_coefficient = permanent / fallback_np.sqrt(
            fallback_np.prod(factorial(output))
        )

        state_vector = connector.assign(state_vector, idx, state_vector_coefficient)

    state_vector /= fallback_np.sqrt(fallback_np.prod(factorial(input)))

    return state_vector


def calculate_inner_product(interferometer, input, output, connector):
    np = connector.np
    fallback_np = connector.fallback_np
    permanent = connector.permanent(interferometer, cols=input, rows=output)

    return permanent / np.sqrt(
        fallback_np.prod(factorial(output)) * fallback_np.prod(factorial(input))
    )


def generate_lossless_samples(input, shots, permanent, interferometer, rng):
    """
    Generates samples corresponding to the Clifford & Clifford algorithm B from
    https://arxiv.org/abs/1706.01260.

    Args:
        input: The input Fock basis state.
        shots: Number of samples to be generated.
        permanent: Permanent calculator function, already containing
            the unitary matrix corresponding to the interferometer.
        rng: Random number generator.

    Returns:
        The generated samples.
    """

    return _generate_samples(
        input,
        shots,
        permanent,
        sample_generator=_generate_lossless_sample,
        interferometer=interferometer,
        rng=rng,
    )


def generate_uniform_lossy_samples(
    input, shots, permanent, interferometer, transmissivity, rng
):
    """
    Basically the same algorithm as in `generate_lossless_samples`, but rejects
    particles according to the uniform `transmissivity` specified.
    """
    sample_generator = partial(
        _generate_uniform_lossy_sample, transmissivity=transmissivity
    )

    return _generate_samples(
        input,
        shots,
        permanent,
        interferometer=interferometer,
        sample_generator=sample_generator,
        rng=rng,
    )


def generate_lossy_samples(
    input_state, samples_number, permanent, interferometer_svd, rng
):
    """
    Basically the same algorithm as in `generate_lossless_samples`, but doubles the
    system size and embeds the input state and the input matrix in order to simulate
    non-uniform losses characterized by the singular values in `interferometer_svd`.
    """
    expanded_matrix = _prepare_interferometer_matrix_in_expanded_space(
        interferometer_svd
    )

    expansion_zeros = np.zeros_like(input_state, dtype=int)
    expanded_state = np.vstack([input_state, expansion_zeros])
    expanded_state = expanded_state.reshape(
        2 * len(input_state),
    )

    expanded_samples = generate_lossless_samples(
        expanded_state, samples_number, permanent, expanded_matrix, rng
    )

    # Trim output state
    samples = []

    for output_state in expanded_samples:
        while len(output_state) > len(input_state):
            output_state = np.delete(output_state, len(output_state) - 1)
        samples.append(output_state)

    return samples


def _get_first_quantized(occupation_numbers):
    first_quantized = np.empty(sum(occupation_numbers), dtype=occupation_numbers.dtype)

    idx = 0
    for mode, reps in enumerate(occupation_numbers):
        for _ in range(reps):
            first_quantized[idx] = mode
            idx += 1

    return first_quantized


def _generate_samples(input, shots, permanent, interferometer, sample_generator, rng):
    d = len(input)
    n = np.sum(input)

    samples = []

    first_quantized_input = _get_first_quantized(input)

    while len(samples) < shots:
        sample = sample_generator(
            d,
            n,
            permanent,
            interferometer,
            first_quantized_input,
            rng=rng,
        )
        samples.append(sample)

    return samples


def _grow_current_input(current_input, to_shrink, rng):
    random_index = rng.choice(len(to_shrink))
    random_mode = to_shrink[random_index]

    current_input[random_mode] += 1

    to_shrink = np.delete(to_shrink, random_index)

    return current_input, to_shrink


def _generate_lossless_sample(
    d, n, permanent, interferometer, first_quantized_input, rng
):
    sample = np.zeros(d, dtype=int)

    current_input = np.zeros(d, dtype=int)
    to_shrink = np.copy(first_quantized_input)

    for _ in range(1, n + 1):
        current_input, to_shrink = _grow_current_input(current_input, to_shrink, rng)

        pmf = _calculate_pmf(current_input, sample, permanent, interferometer)

        index = _sample_from_pmf(pmf, rng)

        sample[index] += 1

    return sample


def _generate_uniform_lossy_sample(
    d, n, permanent, interferometer, first_quantized_input, transmissivity, rng
):
    sample = np.zeros(d, dtype=int)

    number_of_particle_to_sample = 0

    current_input = np.zeros(d, dtype=int)
    to_shrink = np.copy(first_quantized_input)

    for _ in range(1, n + 1):
        if rng.uniform() > transmissivity:
            continue
        else:
            number_of_particle_to_sample += 1

        current_input, to_shrink = _grow_current_input(current_input, to_shrink, rng)

        pmf = _calculate_pmf(current_input, sample, permanent, interferometer)

        index = _sample_from_pmf(pmf, rng)

        sample[index] += 1

    return sample


def _filter_zeros(matrix, input_state, output_state):
    nonzero_input_indices = input_state > 0
    nonzero_output_indices = output_state > 0

    new_input_state = input_state[input_state > 0]
    new_output_state = output_state[output_state > 0]

    new_matrix = matrix[np.ix_(nonzero_output_indices, nonzero_input_indices)]

    return new_matrix, new_input_state, new_output_state


def _calculate_pmf(input_state, sample, calculate_permanent, interferometer):
    d = len(sample)
    pmf = np.empty(d, dtype=np.float64)

    nonzero_indices = np.arange(d)[input_state > 0]

    partial_permanents = np.zeros(len(nonzero_indices), dtype=interferometer.dtype)

    for idx, nonzero_idx in enumerate(nonzero_indices):
        input_state_subtracted = input_state.copy()
        input_state_subtracted[nonzero_idx] -= 1

        filtered_interferometer, filtered_input_state, filtered_output_state = (
            _filter_zeros(interferometer, input_state_subtracted, sample)
        )
        partial_permanents[idx] = calculate_permanent(
            filtered_interferometer, filtered_output_state, filtered_input_state
        )

    normalization = 0.0
    for idx in range(d):
        permanent = 0.0
        for j in range(len(partial_permanents)):
            permanent += (
                input_state[nonzero_indices[j]]
                * partial_permanents[j]
                * interferometer[idx, nonzero_indices[j]]
            )

        pmf[idx] = np.abs(permanent) ** 2
        normalization += pmf[idx]

    return pmf / normalization


def _sample_from_pmf(pmf, rng):
    return rng.choice(np.arange(pmf.shape[0]), p=pmf)


def _prepare_interferometer_matrix_in_expanded_space(interferometer_svd):
    v_matrix, singular_values, u_matrix = interferometer_svd

    expansions_zeros = np.zeros_like(v_matrix)
    expansions_ones = np.eye(len(v_matrix))
    expanded_v = np.block(
        [[v_matrix, expansions_zeros], [expansions_zeros, expansions_ones]]
    )
    expanded_u = np.block(
        [[u_matrix, expansions_zeros], [expansions_zeros, expansions_ones]]
    )
    singular_values_matrix_expansion = _calculate_singular_values_matrix_expansion(
        singular_values
    )
    singular_values_expanded_matrix = np.block(
        [
            [np.diag(singular_values), singular_values_matrix_expansion],
            [singular_values_matrix_expansion, np.diag(singular_values)],
        ]
    )
    return expanded_v @ singular_values_expanded_matrix @ expanded_u


def _calculate_singular_values_matrix_expansion(singular_values_vector):
    vector_of_squared_expansions = 1.0 - np.power(singular_values_vector, 2)

    expansion_values = np.sqrt(vector_of_squared_expansions)

    return np.diag(expansion_values)
