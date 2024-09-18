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
from scipy.special import factorial, binom

from collections import defaultdict
from functools import lru_cache, partial


from piquasso._math.combinatorics import partitions

"""
This is a contribution from `theboss`, see https://github.com/Tomev-CTP/theboss.

The original code has been re-implemented and adapted to Piquasso.
"""

__author__ = "Tomasz Rybotycki"


def calculate_state_vector(interferometer, initial_state, config, calculator):
    """
    Calculates the state vector on the particle subspace defined by `initial_state`.
    """

    np = calculator.np
    fallback_np = calculator.fallback_np

    possible_outputs = partitions(
        particles=calculator.fallback_np.sum(initial_state),
        boxes=len(initial_state),
    )

    state_vector = np.empty(possible_outputs.shape[0], dtype=config.complex_dtype)

    input = initial_state

    for idx, output in enumerate(possible_outputs):
        permanent = calculator.permanent(interferometer, cols=input, rows=output)

        state_vector_coefficient = permanent / fallback_np.sqrt(
            fallback_np.prod(factorial(output))
        )

        state_vector = calculator.assign(state_vector, idx, state_vector_coefficient)

    state_vector /= fallback_np.sqrt(fallback_np.prod(factorial(input)))

    return state_vector


def calculate_inner_product(interferometer, input, output, calculator):
    np = calculator.np
    fallback_np = calculator.fallback_np
    permanent = calculator.permanent(interferometer, cols=input, rows=output)

    return permanent / np.sqrt(
        fallback_np.prod(factorial(output)) * fallback_np.prod(factorial(input))
    )


def generate_lossless_samples(input, shots, calculate_permanent, rng):
    """
    Generates samples corresponding to the Clifford & Clifford algorithm
    from [Brod, Oszmaniec 2020] see
    `this article <https://arxiv.org/abs/1906.06696>`_ for more details.

    Args:
        input: The input Fock basis state.
        shots: Number of samples to be generated.
        calculate_permanent: Permanent calculator function, already containing
            the unitary matrix corresponding to the interferometer.
        rng: Random number generator.

    Returns:
        The generated samples.
    """

    return _generate_samples(
        input,
        shots,
        calculate_permanent,
        sample_generator=_generate_lossless_sample,
        rng=rng,
    )


def generate_uniform_lossy_samples(
    input, shots, calculate_permanent, transmissivity, rng
):
    """
    Basically the same algorithm as in `generate_lossless_samples`, but rejects
    particles according to the uniform `transmissivity` specified.
    """
    sample_generator = partial(
        _generate_uniform_lossy_sample, transmissivity=transmissivity
    )

    return _generate_samples(
        input, shots, calculate_permanent, sample_generator=sample_generator, rng=rng
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

    calculate_permanent = partial(permanent, matrix=expanded_matrix)
    expanded_samples = generate_lossless_samples(
        expanded_state, samples_number, calculate_permanent, rng
    )

    # Trim output state
    samples = []

    for output_state in expanded_samples:
        while len(output_state) > len(input_state):
            output_state = np.delete(output_state, len(output_state) - 1)
        samples.append(output_state)

    return samples


def _generate_samples(input, shots, calculate_permanent, sample_generator, rng):
    d = len(input)
    n = np.sum(input)

    # Labeling possible input states into dict where keys are being number of
    # particles in the state.
    labeled_input_states = _generate_labeled_smaller_arrays(input)

    calculate_weights = _get_weight_calculator(
        input,
        labeled_input_states,
    )

    cache = {
        "all_possible_outputs": dict(),
        "probability_mass_functions": dict(),
    }

    samples = []

    while len(samples) < shots:
        sample, cache = sample_generator(
            d,
            n,
            calculate_permanent,
            calculate_weights,
            labeled_input_states,
            cache,
            rng=rng,
        )
        samples.append(sample)

    return samples


def _generate_lossless_sample(
    d, n, calculate_permanent, calculate_weights, labeled_input_states, cache, rng
):
    sample = np.zeros(d, dtype=int)

    for number_of_particle_to_sample in range(1, n + 1):
        key = tuple(sample)

        if key not in cache["all_possible_outputs"]:
            possible_inputs = labeled_input_states[number_of_particle_to_sample]

            weights = calculate_weights(number_of_particle_to_sample)

            possible_outputs = _generate_possible_output_states(sample)
            pmf = _calculate_pmf(
                possible_inputs, possible_outputs, weights, calculate_permanent
            )

            cache["all_possible_outputs"][key] = possible_outputs
            cache["probability_mass_functions"][key] = pmf
        else:
            possible_outputs = cache["all_possible_outputs"][key]
            pmf = cache["probability_mass_functions"][key]

        sample_index = _sample_from_pmf(pmf, rng)

        sample = possible_outputs[sample_index]

    return sample, cache


def _generate_uniform_lossy_sample(
    d,
    n,
    calculate_permanent,
    calculate_weights,
    labeled_input_states,
    cache,
    transmissivity,
    rng,
):
    sample = np.zeros(d, dtype=int)

    number_of_particle_to_sample = 0

    for _ in range(1, n + 1):
        key = tuple(sample)

        if rng.uniform() > transmissivity:
            continue
        else:
            number_of_particle_to_sample += 1

        if key not in cache["all_possible_outputs"]:
            possible_inputs = labeled_input_states[number_of_particle_to_sample]

            weights = calculate_weights(number_of_particle_to_sample)

            possible_outputs = _generate_possible_output_states(sample)
            pmf = _calculate_pmf(
                possible_inputs, possible_outputs, weights, calculate_permanent
            )

            cache["all_possible_outputs"][key] = possible_outputs
            cache["probability_mass_functions"][key] = pmf
        else:
            possible_outputs = cache["all_possible_outputs"][key]
            pmf = cache["probability_mass_functions"][key]

        sample_index = _sample_from_pmf(pmf, rng)

        sample = possible_outputs[sample_index]

    return sample, cache


def _calculate_pmf(
    possible_inputs,
    possible_outputs,
    weights,
    calculate_permanent,
):
    normalization = 0.0

    no_of_possible_outputs = possible_outputs.shape[0]
    no_of_possible_inputs = len(possible_inputs)

    pmf = np.empty(no_of_possible_outputs, dtype=np.float64)

    for i in range(no_of_possible_outputs):
        inner_sum = 0.0
        possible_output = possible_outputs[i]

        for j in range(no_of_possible_inputs):
            possible_input = possible_inputs[j]

            permanent = calculate_permanent(cols=possible_input, rows=possible_output)
            probability = weights[j] * np.abs(permanent) ** 2
            inner_sum += probability

        normalization += inner_sum
        pmf[i] = inner_sum

    return pmf / normalization


def _sample_from_pmf(pmf, rng):
    return rng.choice(np.arange(pmf.shape[0]), p=pmf)


def _generate_smaller_arrays(array):
    """Generates elementwise smaller non-negative integer arrays from `array`."""
    ranges = [np.arange(x + 1) for x in array]
    grids = np.meshgrid(*ranges, indexing="ij")
    combinations = np.stack(grids, axis=-1).reshape(-1, len(array))
    return combinations


def _generate_labeled_smaller_arrays(array):
    smaller_arrays = _generate_smaller_arrays(array)

    labeled_arrays = defaultdict(list)

    for small_array in smaller_arrays:
        labeled_arrays[np.sum(small_array)].append(small_array)

    for key in labeled_arrays.keys():
        labeled_arrays[key] = np.array(labeled_arrays[key])

    return labeled_arrays


def _generate_possible_output_states(sample):
    d = sample.shape[0]

    repeated_sample = np.repeat(sample[None, :], d, axis=0)

    return repeated_sample + np.identity(d, dtype=sample.dtype)


def _get_weight_calculator(input_state, labeled_states):
    @lru_cache
    def _calculate_weights(number_of_particle_to_sample):
        n = np.sum(input_state)

        fac_no_of_particles = factorial(number_of_particle_to_sample)

        possible_input_states = labeled_states[number_of_particle_to_sample]

        factorials = (
            np.prod(factorial(possible_input_states), axis=1) * fac_no_of_particles
        )

        size = len(possible_input_states)

        weights = np.empty(shape=size, dtype=np.float64)
        normalization = 0.0

        for i in range(size):
            weight = np.prod(binom(input_state, possible_input_states[i]))
            normalization += weight
            weights[i] = weight

        divisor = normalization * binom(n, number_of_particle_to_sample) * factorials

        return weights / divisor

    return _calculate_weights


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
