#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
from scipy.special import factorial, comb

from functools import partial

from piquasso.api.exceptions import InvalidSimulation

from piquasso._math.fock import get_fock_space_basis, get_postselected_fock_basis
from piquasso._math.polynomial import multiply_by_linear_truncated
from .marginal import (
    get_binomial_moments,
    get_single_marginal_probability_from_binomial_moments,
)
from piquasso._math.indices import to_first_quantized, to_second_quantized

from .probabilities import get_lossy_partially_distinguishable_detection_probabilities


def generate_samples(
    input,
    shots,
    calculate_permanent_laplace,
    interferometer,
    reject_condition,
    postselect_data,
    uniform_particle_overlap,
    config,
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
        reject_condition: A callable that returns True if a particle should be rejected
            during the sample generation.
        postselect_data: A tuple containing postselection information:
            - postselect_modes: Tuple of modes where postselection is applied.
            - postselect_photons: Tuple of number of photons to postselect in the
              corresponding modes.
            - max_sample_generation_trials: Maximum number of trials to generate a valid
              sample, to avoid infinite loops in case the postselection conditions are
              too strict.
        uniform_particle_overlap: Uniform particle overlap parameter for partially
            distinguishable photons, or None for fully indistinguishable photons.
        config: The simulator configuration, providing the random number generator and
            other settings.

    Returns:
        The generated samples.
    """

    is_postselected = len(postselect_data[0]) > 0

    if is_postselected:
        if uniform_particle_overlap is None:
            sample_generator = partial(
                _generate_sample_with_postselect,
                reject_condition=reject_condition,
                postselect_data=postselect_data,
            )
        else:
            sample_generator = partial(
                _generate_sample_with_postselect_and_uniform_overlap,
                reject_condition=reject_condition,
                postselect_data=postselect_data,
                uniform_particle_overlap=uniform_particle_overlap,
            )
    else:
        if uniform_particle_overlap is None:
            sample_generator = partial(
                _generate_sample, reject_condition=reject_condition
            )
        else:
            sample_generator = partial(
                _generate_sample_with_uniform_overlap,
                reject_condition=reject_condition,
                uniform_particle_overlap=uniform_particle_overlap,
            )

    return _generate_samples(
        input,
        shots,
        calculate_permanent_laplace,
        sample_generator=sample_generator,
        interferometer=interferometer,
        config=config,
    )


def generate_lossy_samples(
    input,
    shots,
    calculate_permanent_laplace,
    interferometer,
    postselect_data,
    config,
):
    """
    Basically the same algorithm as in `generate_samples`, but doubles the system size
    and embeds the input state and the input matrix in order to simulate non-uniform
    losses characterized by the singular values in `interferometer_svd`.
    """
    interferometer_svd = np.linalg.svd(interferometer)
    expanded_matrix = _prepare_interferometer_matrix_in_expanded_space(
        interferometer_svd
    )

    expansion_zeros = np.zeros_like(input, dtype=int)
    expanded_state = np.vstack([input, expansion_zeros])
    expanded_state = expanded_state.reshape(2 * len(input))

    expanded_samples = generate_samples(
        expanded_state,
        shots,
        calculate_permanent_laplace,
        expanded_matrix,
        reject_condition=lambda: False,
        postselect_data=postselect_data,
        uniform_particle_overlap=None,
        config=config,
    )

    # Trim output state
    samples = [x[: len(input)] for x in expanded_samples]

    return samples


def _generate_samples(
    input, shots, calculate_permanent_laplace, interferometer, sample_generator, config
):
    d = len(input)
    n = np.sum(input)

    samples = []

    first_quantized_input = to_first_quantized(input)

    def _generate_sample_from_seed(seed):
        rng = np.random.default_rng(seed=seed)
        return tuple(
            sample_generator(
                d,
                n,
                calculate_permanent_laplace,
                interferometer,
                first_quantized_input,
                rng=rng,
            )
        )

    seed = config.seed_sequence

    if config.use_dask:
        try:
            import dask
        except ImportError:
            raise ImportError("This feature requires 'dask' to be installed.")

        delayed_func = dask.delayed(_generate_sample_from_seed)

        compute_list = []
        for idx in range(shots):
            compute_list.append(delayed_func(seed=seed + idx))

        results = dask.compute(*compute_list)

        return list(results)

    for idx in range(shots):
        sample = _generate_sample_from_seed(seed=seed + idx)
        samples.append(sample)

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
    track_photons_needed=True,
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
                diff[tuple(postselect_modes).index(index)] -= 1

                if np.any(diff < 0):
                    break

            if track_photons_needed and photons_needed > n - k:
                break

        else:
            retry = False

    if not track_photons_needed:
        return sample

    trimmed_sample = np.delete(sample, postselect_modes)

    return trimmed_sample


def _generate_sample_with_uniform_overlap(
    d,
    n,
    calculate_permanent_laplace,
    interferometer,
    first_quantized_input,
    rng,
    reject_condition,
    uniform_particle_overlap,
):
    """Algorithm for generating samples from a lossy and partially distinguishable
    system with uniform particle overlap. The algorithm separates the particles into
    indistinguishable and distinguishable ones, generates a sample for the
    indistinguishable particles, and then generates a sample for the distinguishable
    particles. The samples are then combined to form the final output sample.

    Source: Renema et al., "Efficient Classical Algorithm for Boson Sampling with
    Partially Distinguishable Photons", Phys. Rev. Lett. 120, 220502 (2018),
    https://arxiv.org/abs/1707.02793.
    """
    second_quantized_input = to_second_quantized(first_quantized_input, d)

    indist_particles, dist_particles = _separate_particles(
        second_quantized_input, uniform_particle_overlap, rng
    )

    dist_output = _sample_distinguishable_particles(
        interferometer, dist_particles, rng, reject_condition
    )

    indist_output = _generate_sample(
        d,
        np.sum(indist_particles),
        calculate_permanent_laplace=calculate_permanent_laplace,
        interferometer=interferometer,
        first_quantized_input=to_first_quantized(indist_particles),
        rng=rng,
        reject_condition=reject_condition,
    )

    return tuple(dist_output + indist_output)


def _separate_particles(input_occupation, uniform_particle_overlap, rng):
    d = len(input_occupation)

    indist_particles = np.zeros(d, dtype=int)
    dist_particles = np.zeros(d, dtype=int)

    for j, n_j in enumerate(input_occupation):
        if n_j == 0:
            continue

        weights = np.array(
            [
                comb(n_j, k)
                * (uniform_particle_overlap**k)
                * ((1.0 - uniform_particle_overlap) ** (n_j - k))
                * factorial(k)
                for k in range(n_j + 1)
            ],
            dtype=float,
        )

        probabilities = weights / weights.sum()

        K_j = rng.choice(n_j + 1, p=probabilities)

        indist_particles[j] = K_j
        dist_particles[j] = n_j - K_j

    return indist_particles, dist_particles


def _sample_distinguishable_particles(interferometer, particles, rng, reject_condition):
    d = len(particles)
    output = np.zeros(d, dtype=int)

    for input_mode, count in enumerate(particles):
        probabilities = np.abs(interferometer[:, input_mode]) ** 2

        # NOTE: This normalization is necessary because the interferometer may not be
        # unitary in the case of losses.
        normalized_probabilities = probabilities / np.sum(probabilities)

        for _ in range(count):
            output_mode = rng.choice(d, p=normalized_probabilities)

            if not reject_condition():
                output[output_mode] += 1

    return output


def _generate_sample_with_postselect_and_uniform_overlap(
    d,
    n,
    calculate_permanent_laplace,
    interferometer,
    first_quantized_input,
    rng,
    reject_condition,
    postselect_data,
    uniform_particle_overlap,
):
    """
    Greedy algorithm to generate samples from a lossy and partially distinguishable
    system with postselection. The algorithm separates the particles into
    indistinguishable and distinguishable ones, generates a sample for the
    indistinguishable particles, and then generates a sample for the distinguishable
    particles conditioned on the postselection of the indistinguishable particles. If
    the postselection fails, the algorithm retries until a valid sample is generated or
    the maximum number of trials is reached, similarly to
    `_generate_sample_with_postselect`.

    NOTE: This might not be the best option here, but maybe good enough for small
    systems. To avoid the possibility of infinite loops and a blow-up in runtime, one
    might be tempted to implement this as a mode-by-mode strategy, but computing the
    marginal probabilities for the distinguishable particles is not trivial, and it is
    not clear how to do it efficiently. Therefore, we stick to the greedy approach for
    now. It is also possible to just use the naive
    `generate_lossy_and_partially_distinguishable_samples`, which is simpler, but may be
    slower for larger systems and if `shots` is not too large.
    """
    postselect_modes, postselect_photons, max_sample_generation_trials = postselect_data

    postselect_modes = np.asarray(postselect_modes, dtype=int)
    postselect_photons = np.asarray(postselect_photons, dtype=int)

    second_quantized_input = to_second_quantized(first_quantized_input, d)

    trial_count = 0

    while True:
        trial_count += 1

        if trial_count > max_sample_generation_trials:
            raise InvalidSimulation(
                "Too many trials during sample generation. "
                "The specified postselection criteria may be very highly unlikely. "
                "Aborting. "
                "To increase the limit, set Config.max_sample_generation_trials "
                f"to a higher value. Current value: {max_sample_generation_trials}."
            )

        indist_particles, dist_particles = _separate_particles(
            second_quantized_input,
            uniform_particle_overlap,
            rng,
        )

        indist_number_of_particles = np.sum(indist_particles)
        indist_first_quantized_input = to_first_quantized(indist_particles)

        try:
            indist_output = _generate_sample_with_postselect(
                d,
                indist_number_of_particles,
                calculate_permanent_laplace=calculate_permanent_laplace,
                interferometer=interferometer,
                first_quantized_input=indist_first_quantized_input,
                rng=rng,
                reject_condition=reject_condition,
                postselect_data=(postselect_modes, postselect_photons, 1),
                track_photons_needed=False,
            )
        except InvalidSimulation:
            continue

        postselect_photons_remaining = (
            postselect_photons - indist_output[postselect_modes]
        )

        dist_postselection_probability = _calculate_dist_postselection_probability(
            interferometer=interferometer,
            dist_particles=dist_particles,
            postselect_modes=postselect_modes,
            postselect_photons=postselect_photons_remaining,
        )

        if rng.random() > dist_postselection_probability:
            continue

        probability_table = _calculate_dist_postselection_probability_table(
            interferometer=interferometer,
            dist_particles=dist_particles,
            postselect_modes=postselect_modes,
            postselect_photons_bound=postselect_photons,
        )

        output = _sample_dist_output_conditioned_on_postselection(
            interferometer=interferometer,
            dist_particles=dist_particles,
            postselect_modes=postselect_modes,
            postselect_photons=postselect_photons_remaining,
            probability_table=probability_table,
            rng=rng,
        )

        output = output + indist_output

        return tuple(np.delete(output, postselect_modes))


def _calculate_dist_postselection_probability(
    interferometer,
    dist_particles,
    postselect_modes,
    postselect_photons,
):
    float_dtype = np.finfo(interferometer.dtype).dtype

    polynomial = np.zeros(tuple(postselect_photons + 1), dtype=float_dtype)
    polynomial[(0,) * len(postselect_photons)] = 1.0

    for input_mode, multiplicity in enumerate(dist_particles):
        probabilities = np.abs(interferometer[postselect_modes, input_mode]) ** 2

        for _ in range(multiplicity):
            multiply_by_linear_truncated(
                polynomial, 1.0 - probabilities.sum(), probabilities, out=polynomial
            )

    return polynomial[tuple(postselect_photons)]


def _calculate_dist_postselection_probability_table(
    interferometer,
    dist_particles,
    postselect_modes,
    postselect_photons_bound,
):
    dist_input = to_first_quantized(dist_particles)
    float_dtype = np.finfo(interferometer.dtype).dtype

    probability_table = [
        np.zeros(tuple(postselect_photons_bound + 1), dtype=float_dtype)
        for _ in range(len(dist_input) + 1)
    ]

    probability_table[-1][(0,) * len(postselect_photons_bound)] = 1.0

    for photon_index in range(len(dist_input) - 1, -1, -1):
        input_mode = dist_input[photon_index]

        probabilities = np.abs(interferometer[postselect_modes, input_mode]) ** 2

        multiply_by_linear_truncated(
            probability_table[photon_index + 1],
            1.0 - probabilities.sum(),
            probabilities,
            out=probability_table[photon_index],
        )

    return probability_table


def _sample_dist_output_conditioned_on_postselection(
    interferometer,
    dist_particles,
    postselect_modes,
    postselect_photons,
    probability_table,
    rng,
):
    d = interferometer.shape[0]
    float_dtype = np.finfo(interferometer.dtype).dtype

    dist_input = to_first_quantized(dist_particles)
    non_postselect_modes = np.delete(np.arange(d), postselect_modes)

    sample = np.zeros(d, dtype=int)
    remaining = np.array(postselect_photons, dtype=int)

    for photon_index, input_mode in enumerate(dist_input):
        next_table = probability_table[photon_index + 1]

        future_probability = next_table[tuple(remaining)]

        non_postselect_probabilities = (
            np.abs(interferometer[non_postselect_modes, input_mode]) ** 2
        )

        postselect_probabilities = (
            np.abs(interferometer[postselect_modes, input_mode]) ** 2
        )

        loss_probability = (
            1.0 - non_postselect_probabilities.sum() - postselect_probabilities.sum()
        )

        non_postselect_weights = non_postselect_probabilities * future_probability
        loss_weight = loss_probability * future_probability

        postselect_weights = np.zeros(len(postselect_modes), dtype=float_dtype)

        for axis, probability in enumerate(postselect_probabilities):
            if remaining[axis] == 0:
                continue

            remaining[axis] -= 1
            postselect_weights[axis] = probability * next_table[tuple(remaining)]
            remaining[axis] += 1

        weights = np.concatenate(
            [
                non_postselect_weights,
                np.array([loss_weight], dtype=float_dtype),
                postselect_weights,
            ]
        )

        weights /= np.sum(weights)

        index = rng.choice(len(weights), p=weights)

        if index < len(non_postselect_modes):
            sample[non_postselect_modes[index]] += 1

        elif index == len(non_postselect_modes):
            pass

        else:
            axis = index - len(non_postselect_modes) - 1
            sample[postselect_modes[axis]] += 1
            remaining[axis] -= 1

    return sample


def generate_lossy_and_partially_distinguishable_samples(
    input,
    shots,
    interferometer,
    postselect_data,
    particle_overlap_matrix,
    connector,
    config,
):
    """
    Naive algorithm to generate samples from a general lossy and partially
    distinguishable system with postselection.

    NOTE: This might not be the best option here, but good enough for small systems,
    especially if `shots` is large.
    """
    postselect_modes = postselect_data[0]
    postselect_photons = postselect_data[1]

    occupation_numbers = get_postselected_fock_basis(
        d=len(input),
        cutoff=np.sum(input) + 1,
        postselected_modes=postselect_modes,
        postselected_photons=postselect_photons,
    )
    if len(postselect_modes) > 0:
        trimmed_occupation_numbers = np.delete(
            occupation_numbers, postselect_modes, axis=1
        )
    else:
        trimmed_occupation_numbers = occupation_numbers

    probabilities = get_lossy_partially_distinguishable_detection_probabilities(
        occupation_numbers=occupation_numbers,
        transmission_matrix=interferometer,
        input_occupation=input,
        particle_overlap=particle_overlap_matrix,
        connector=connector,
    )

    probabilities /= connector.np.sum(probabilities)

    sample_indices = config.rng.choice(
        len(occupation_numbers),
        size=shots,
        p=probabilities,
    )

    samples = trimmed_occupation_numbers[sample_indices]

    return [tuple(sample) for sample in samples]


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


def map_to_original_modes(modes, postselected_modes):
    modes = np.asarray(modes, dtype=int).copy()

    for postselected in sorted(postselected_modes):
        modes[modes >= postselected] += 1

    return tuple(modes)


def generate_marginal_samples(
    initial_state, interferometer, modes, shots, rng, postselect_data
):
    n = sum(initial_state)
    k = len(modes)

    postselect_modes = postselect_data[0]
    postselect_photons = np.asarray(postselect_data[1], dtype=int)

    all_modes = np.array(postselect_modes + modes)

    binomial_moments = get_binomial_moments(
        input_photons=initial_state,
        interferometer=interferometer,
        all_modes=all_modes,
        compositions=get_fock_space_basis(len(all_modes), n + 1),
    )

    get_probability = partial(
        get_single_marginal_probability_from_binomial_moments,
        n=n,
        binomial_moments=binomial_moments,
        d=len(all_modes),
    )

    postselection_probability = get_probability(particles=postselect_photons)

    particle_offset = sum(postselect_photons)
    mode_offset = len(postselect_photons)

    samples = []

    for _ in range(shots):
        previous_probability = postselection_probability
        remaining_particles = n - particle_offset

        particles = np.concatenate([postselect_photons, np.zeros(k, dtype=int)])

        for mode_idx in range(k):
            threshold = rng.random()

            cumulative_probability = 0.0

            total_mode_idx = mode_offset + mode_idx

            for photon_number in range(remaining_particles + 1):
                particles[total_mode_idx] = photon_number

                probability = get_probability(particles=particles[: total_mode_idx + 1])

                conditional_probability = probability / previous_probability
                cumulative_probability += conditional_probability

                if cumulative_probability >= threshold:
                    break

            remaining_particles -= photon_number
            previous_probability = probability

        samples.append(tuple(particles[mode_offset:]))

    return samples


def is_direct_marginal_sampling_cheaper(k: int, d: int, n: int, shots: int) -> bool:
    """
    Estimate whether direct marginal sampling is cheaper than full sampling followed by
    discarding unmeasured modes. The decision is based on the size of the number of
    coefficients in the marginal sampling algorithm and the number of shots.

    NOTE: Take this with a grain of salt, as it is based on rough guesses and benchmarks
    on my laptop, and the specific runtime may vary depending on the machine and the
    specific parameters.
    """

    if k == d:
        return False

    number_of_coeffs_in_marginal_sampling_algorithm = sum(
        comb(degree + k - 1, k - 1) ** 2 for degree in range(n + 1)
    )

    if number_of_coeffs_in_marginal_sampling_algorithm <= 100_000:
        return True

    if number_of_coeffs_in_marginal_sampling_algorithm <= 500_000:
        return shots >= 1_000

    if number_of_coeffs_in_marginal_sampling_algorithm <= 4_000_000:
        return shots >= 10_000

    return False
