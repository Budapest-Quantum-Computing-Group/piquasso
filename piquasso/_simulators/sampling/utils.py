#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

from piquasso.api.exceptions import InvalidSimulation

from piquasso._math.combinatorics import partitions, partitions_bounded_k

"""
This is a contribution from `theboss`, see https://github.com/Tomev-CTP/theboss.

The original code has been re-implemented and adapted to Piquasso.
"""

__author__ = "Tomasz Rybotycki"


def calculate_state_vector(
    interferometer,
    initial_state,
    postselect_data,
    config,
    connector,
):
    """Calculate the state vector on the particle subspace defined by ``initial_state``.

    This implementation follows Algorithm 1 (``SLOS_full``) from
    `Strong Simulation of Linear Optical Processes`.

    The algorithm is modified by pruning to account for postselection on certain modes.
    """

    np = connector.np
    fallback_np = connector.fallback_np

    is_postselected = len(postselect_data[0]) > 0

    d = len(initial_state)
    n = int(fallback_np.sum(initial_state))

    if is_postselected:
        """
        The idea is the following:

        Let's say that we want to do a full state vector simulation on 3 modes for 4
        photons, but we only want the components in the state vector for which the
        particle number in the last two modes are, e.g., (1, 1). So, for us, all the
        interesting Fock basis states are |2, 0, 1, 1>, |1, 1, 1, 1>, |0, 2, 1, 1>. In
        the k_limit=0 case, the `partitions_bounded_k` function calculates these.

        However, let's say that you're calculating the state vector with SLOS, meaning
        that you are calculating the state vector particle-by-particle. Then of course,
        you cannot have a transition, e.g., |2, 1, 0, 0> -> |x, y, 1, 1> (where x+y=2),
        so you would like to disregard |2, 1, 0, 0>, but for example |1, 1, 0, 1> would
        be fine. More specifically, the condition is to let the difference on the
        postselected modes be up to some number k_limit, where we will set k_limit
        according to the number of particles we still need to add to the state vector
        (because they could end up in the postselected modes still).
        """
        postselect_modes, postselect_photons = postselect_data
        bases = [
            partitions_bounded_k(
                boxes=d,
                particles=k,
                constrained_boxes=postselect_modes,
                max_per_box=postselect_photons,
                k_limit=(n - k),
            )
            for k in range(n + 1)
        ]
    else:
        bases = [partitions(boxes=d, particles=k) for k in range(n + 1)]

    index_maps = [{tuple(basis[i]): i for i in range(len(basis))} for basis in bases]

    sigma = fallback_np.zeros(d, dtype=int)
    schedule_sigma = []
    schedule_mode = []
    for p in range(d):
        for _ in range(initial_state[p]):
            schedule_sigma.append(sigma.copy())
            schedule_mode.append(p)
            sigma[p] += 1

    UF_curr = np.zeros(len(bases[0]), dtype=config.complex_dtype)
    UF_curr = connector.assign(UF_curr, 0, 1.0)

    for k in range(n):
        sigma_k = schedule_sigma[k]
        p = schedule_mode[k]
        index_map = index_maps[k + 1]
        UF_next = np.zeros(len(bases[k + 1]), dtype=config.complex_dtype)

        for idx_t, t in enumerate(bases[k]):
            A = UF_curr[idx_t]

            for i in range(d):
                t_i = t[i]
                new_t = t.copy()
                new_t[i] += 1
                tuple_new_t = tuple(new_t)
                if tuple_new_t not in index_map:
                    continue

                index_new = index_map[tuple_new_t]

                factor = fallback_np.sqrt((t_i + 1) / (sigma_k[p] + 1))

                UF_next = connector.assign(
                    UF_next,
                    index_new,
                    UF_next[index_new] + factor * interferometer[i, p] * A,
                )

        UF_curr = UF_next

    return UF_curr


def calculate_inner_product(interferometer, input, output, connector):
    np = connector.np
    fallback_np = connector.fallback_np
    permanent = connector.permanent(interferometer, cols=input, rows=output)

    return permanent / np.sqrt(
        fallback_np.prod(factorial(output)) * fallback_np.prod(factorial(input))
    )


def generate_samples(
    input,
    shots,
    calculate_permanent_laplace,
    interferometer,
    rng,
    reject_condition,
    postselect_data,
):
    """
    Generates samples corresponding to the Clifford & Clifford algorithm B from
    https://arxiv.org/abs/1706.01260.

    Args:
        input: The input Fock basis state.
        shots: Number of samples to be generated.
        calculate_permanent_laplace: Function to calculate the permanent submatrices
            according to the Laplace expansion.
        interferometer: The interferometer matrix.
        rng: Random number generator.
        reject_condition: A callable that returns True if a particle should be rejected
            during the sample generation.
        postselect_data: A tuple containing postselection information:
            - postselect_modes: Tuple of modes where postselection is applied.
            - postselect_photons: Tuple of number of photons to postselect in the
              corresponding modes.
            - max_sample_generation_trials: Maximum number of trials to generate a valid
              sample, to avoid infinite loops in case the postselection conditions are
              too strict.

    Returns:
        The generated samples.
    """

    if len(postselect_data[0]) > 0:
        sample_generator = partial(
            _generate_sample_with_postselect,
            reject_condition=reject_condition,
            postselect_data=postselect_data,
        )
    else:
        sample_generator = partial(_generate_sample, reject_condition=reject_condition)

    return _generate_samples(
        input,
        shots,
        calculate_permanent_laplace,
        sample_generator=sample_generator,
        interferometer=interferometer,
        rng=rng,
    )


def generate_lossy_samples(
    input_state,
    samples_number,
    calculate_permanent_laplace,
    interferometer_svd,
    rng,
    postselect_data,
):
    """
    Basically the same algorithm as in `generate_samples`, but doubles the system size
    and embeds the input state and the input matrix in order to simulate non-uniform
    losses characterized by the singular values in `interferometer_svd`.
    """
    expanded_matrix = _prepare_interferometer_matrix_in_expanded_space(
        interferometer_svd
    )

    expansion_zeros = np.zeros_like(input_state, dtype=int)
    expanded_state = np.vstack([input_state, expansion_zeros])
    expanded_state = expanded_state.reshape(2 * len(input_state))

    expanded_samples = generate_samples(
        expanded_state,
        samples_number,
        calculate_permanent_laplace,
        expanded_matrix,
        rng,
        reject_condition=lambda: False,
        postselect_data=postselect_data,
    )

    # Trim output state
    samples = [x[: len(input_state)] for x in expanded_samples]

    return samples


def _get_first_quantized(occupation_numbers):
    first_quantized = np.empty(sum(occupation_numbers), dtype=occupation_numbers.dtype)

    idx = 0
    for mode, reps in enumerate(occupation_numbers):
        for _ in range(reps):
            first_quantized[idx] = mode
            idx += 1

    return first_quantized


def _generate_samples(
    input, shots, calculate_permanent_laplace, interferometer, sample_generator, rng
):
    d = len(input)
    n = np.sum(input)

    samples = []

    first_quantized_input = _get_first_quantized(input)

    while len(samples) < shots:
        sample = sample_generator(
            d,
            n,
            calculate_permanent_laplace,
            interferometer,
            first_quantized_input,
            rng=rng,
        )
        samples.append(tuple(sample))

    return samples


def _grow_current_input(current_input, to_shrink, rng):
    random_index = rng.choice(len(to_shrink))
    random_mode = to_shrink[random_index]

    current_input[random_mode] += 1

    to_shrink = np.delete(to_shrink, random_index)

    return current_input, to_shrink


def _generate_sample(
    d,
    n,
    calculate_permanent_laplace,
    interferometer,
    first_quantized_input,
    rng,
    reject_condition,
):
    sample = np.zeros(d, dtype=int)

    current_input = np.zeros(d, dtype=int)
    to_shrink = np.copy(first_quantized_input)

    for _ in range(1, n + 1):
        if reject_condition():
            continue

        current_input, to_shrink = _grow_current_input(current_input, to_shrink, rng)

        pmf = _calculate_pmf(
            current_input, sample, calculate_permanent_laplace, interferometer
        )

        index = _sample_from_pmf(pmf, rng)

        sample[index] += 1

    return sample


def _generate_sample_with_postselect(
    d,
    n,
    calculate_permanent_laplace,
    interferometer,
    first_quantized_input,
    rng,
    reject_condition,
    postselect_data,
):
    """
    This is a modified version of `_generate_sample` that accounts for postselection.

    The postselection is defined by `postselect_data`, which is a tuple containing:
    - postselect_modes: List of modes where postselection is applied.
    - postselect_photons: List of number of photons to postselect in the corresponding
      modes.
    - max_sample_generation_trials: Maximum number of trials to generate a valid sample,
      to avoid infinite loops in case the postselection conditions are too strict.
    """
    postselect_modes, postselect_photons, max_sample_generation_trials = postselect_data

    retry = True
    trial_count = 0
    photons_needed_orig = np.sum(postselect_photons)

    current_input = np.empty(d, dtype=int)

    sample = np.empty(d, dtype=int)

    while retry:
        trial_count += 1

        if trial_count > max_sample_generation_trials:
            raise InvalidSimulation(
                "Too many trials during sample generation. "
                "The specified postselection criteria may be very highly unlikely. "
                "Aborting. "
                "To increase the limit, set Config.max_sample_generation_trials "
                f"to a higher value. Current value: {max_sample_generation_trials}."
            )

        current_input[:] = 0
        sample[:] = 0

        photons_needed = photons_needed_orig

        diff = np.copy(postselect_photons)

        to_shrink = np.copy(first_quantized_input)

        for k in range(1, n + 1):
            if reject_condition():
                continue

            current_input, to_shrink = _grow_current_input(
                current_input, to_shrink, rng
            )

            pmf = _calculate_pmf(
                current_input, sample, calculate_permanent_laplace, interferometer
            )

            index = _sample_from_pmf(pmf, rng)

            sample[index] += 1

            if index in postselect_modes:
                photons_needed -= 1
                diff[postselect_modes.index(index)] -= 1

                if np.any(diff < 0):
                    break

            if photons_needed > n - k:
                break

        else:
            retry = False

    trimmed_sample = np.delete(sample, postselect_modes)
    return trimmed_sample


def _filter_zeros(matrix, input_state, output_state):
    nonzero_input_indices = input_state > 0
    nonzero_output_indices = output_state > 0

    new_input_state = input_state[nonzero_input_indices]
    new_output_state = output_state[nonzero_output_indices]

    new_matrix = matrix[np.ix_(nonzero_output_indices, nonzero_input_indices)]

    return new_matrix, new_input_state, new_output_state


def _calculate_pmf(input_state, sample, calculate_permanent_laplace, interferometer):
    d = len(sample)
    pmf = np.empty(d, dtype=np.float64)

    nonzero_indices = np.arange(d)[input_state > 0]

    filtered_interferometer, filtered_input_state, filtered_output_state = (
        _filter_zeros(interferometer, input_state, sample)
    )
    partial_permanents = calculate_permanent_laplace(
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
