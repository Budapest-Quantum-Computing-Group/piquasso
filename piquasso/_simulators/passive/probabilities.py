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

from math import comb

from scipy.special import factorial

from piquasso._math.linalg import assym_reduce
from piquasso._math.indices import to_first_quantized
from piquasso._math.polynomial import multiply_by_linear_truncated
from .utils import calculate_inner_product, calculate_lossy_density_matrix_element


def get_ideal_particle_number_probability(
    occupation_number,
    interferometer,
    occupation_numbers,
    coefficients,
    connector,
):
    np = connector.np
    fallback_np = connector.fallback_np

    output_number_of_particles = fallback_np.sum(occupation_number)

    sum_ = 0.0

    for index, input_occupation_number in enumerate(occupation_numbers):
        if output_number_of_particles != fallback_np.sum(input_occupation_number):
            continue

        inner_product = calculate_inner_product(
            interferometer=interferometer,
            input=input_occupation_number,
            output=occupation_number,
            connector=connector,
        )
        coefficient = coefficients[index]

        sum_ += coefficient * inner_product

    return np.abs(sum_) ** 2


def get_lossy_particle_number_probability(
    occupation_number,
    transmission_matrix,
    occupation_numbers,
    coefficients,
    connector,
):
    np = connector.np
    fallback_np = connector.fallback_np

    output_number_of_particles = fallback_np.sum(occupation_number)

    loss_channel_matrix = _build_loss_channel_matrix(
        transmission_matrix=transmission_matrix,
        connector=connector,
    )

    probability = 0.0 + 0.0j

    for left_index, input_left in enumerate(occupation_numbers):
        input_left_number = int(fallback_np.sum(input_left))

        if input_left_number < output_number_of_particles:
            continue

        left_coefficient = coefficients[left_index]

        for right_index, input_right in enumerate(occupation_numbers):
            input_right_number = int(fallback_np.sum(input_right))

            if input_right_number < output_number_of_particles:
                continue

            # For a diagonal output probability, the number of lost photons
            # must be the same on ket and bra sides.
            if input_left_number != input_right_number:
                continue

            right_coefficient = coefficients[right_index]

            matrix_element = calculate_lossy_density_matrix_element(
                input_left=input_left,
                input_right=input_right,
                output_left=occupation_number,
                output_right=occupation_number,
                loss_channel_matrix=loss_channel_matrix,
                connector=connector,
            )

            probability += (
                left_coefficient * np.conj(right_coefficient) * matrix_element
            )

    return np.real_if_close(probability)


def _build_loss_channel_matrix(transmission_matrix, connector):
    """Build A_Phi for a lossy Fock-state channel."""
    fallback_np = connector.fallback_np

    T = fallback_np.asarray(transmission_matrix, dtype=complex)

    d = T.shape[0]

    zero = fallback_np.zeros((d, d), dtype=complex)
    identity = fallback_np.identity(d, dtype=complex)

    T_dagger = T.conj().T

    return fallback_np.block(
        [
            [
                zero,
                T_dagger,
                identity - T_dagger @ T,
                zero,
            ],
            [
                T.conj(),
                zero,
                zero,
                zero,
            ],
            [
                identity - T.T @ T.conj(),
                zero,
                zero,
                T.T,
            ],
            [
                zero,
                zero,
                T,
                zero,
            ],
        ]
    )


def get_partially_distinguishable_detection_probability(
    occupation_number,
    interferometer,
    input_occupation,
    uniform_particle_overlap,
    connector,
):
    """Compute the probability of detecting a given occupation number in a uniformly
    partially distinguishable Boson Sampling experiment, using the rank-3 tensor
    permanent formula.

    See Section III in https://arxiv.org/abs/1410.7687
    """
    fallback_np = connector.fallback_np

    number_of_particles = fallback_np.sum(input_occupation)

    if number_of_particles != fallback_np.sum(occupation_number):
        return 0.0

    # TODO: There must be a faster way to do this when there are collisions.
    A = assym_reduce(interferometer, occupation_number, input_occupation)
    tensor_permanent = _calculate_tensor_permanent(
        A=A,
        particle_overlap=uniform_particle_overlap,
        connector=connector,
    )

    input_norm = _uniform_input_norm(
        input_occupation, uniform_particle_overlap, connector
    )

    denominator = input_norm * fallback_np.prod(factorial(occupation_number))

    return tensor_permanent / denominator


def _calculate_tensor_permanent(A, particle_overlap, connector):
    number_of_particles = A.shape[1]
    number_of_subsets = 1 << number_of_particles

    if number_of_particles == 0:
        return 1.0

    row_sums = connector.np.zeros(
        (number_of_subsets, number_of_particles),
        dtype=A.dtype,
    )

    row_abs_squared_sums = connector.np.zeros(
        (number_of_subsets, number_of_particles),
        dtype=A.dtype,
    )

    abs_A_squared = connector.np.abs(A) ** 2

    for subset in range(1, number_of_subsets):
        least_significant_bit = subset & -subset
        column = least_significant_bit.bit_length() - 1
        previous_subset = subset ^ least_significant_bit

        row_sums[subset] = row_sums[previous_subset] + A[:, column]
        row_abs_squared_sums[subset] = (
            row_abs_squared_sums[previous_subset] + abs_A_squared[:, column]
        )

    tensor_permanent = 0.0

    for subset_s in range(1, 1 << number_of_particles):
        sign_s = -1 if (number_of_particles - subset_s.bit_count()) % 2 else 1

        for subset_t in range(1, 1 << number_of_particles):
            sign_t = -1 if (number_of_particles - subset_t.bit_count()) % 2 else 1

            intersection = subset_s & subset_t

            row_values = (
                particle_overlap
                * row_sums[subset_s]
                * connector.np.conj(row_sums[subset_t])
                + (1.0 - particle_overlap) * row_abs_squared_sums[intersection]
            )

            tensor_permanent += sign_s * sign_t * connector.np.prod(row_values)

    return tensor_permanent


def get_lossy_partially_distinguishable_detection_probabilities(
    occupation_numbers,
    transmission_matrix,
    input_occupation,
    particle_overlap,
    connector,
):
    """Compute the probabilities in a Boson Sampling experiment in general.

    This is not the most efficient way to compute the probabilities, but it is the most
    general. It can handle any kind of partial distinguishability between the particles,
    and kind of loss in the system. It is also good for computing the probabilities for
    many different output occupation numbers at once.

    This uses the coefficient-extraction formula

    P(s) = [x^s] Per(B_loss + sum_m x_m B_m) / Z_in,

    where B_loss = G * (I - T^dagger T) and
    B_m = G * outer(T[m, input_modes], conj(T[m, input_modes])). T is of course the
    transmission matrix of the interferometer, and G is the Gram matrix of the
    particle overlaps.

    The permanent of the polynomial-valued matrix is evaluated using Ryser's
    inclusion-exclusion formula, while the polynomial is truncated to the
    largest requested output occupation.
    """
    np = connector.np
    fallback_np = connector.fallback_np

    complex_dtype = transmission_matrix.dtype

    if occupation_numbers.ndim == 1:
        occupation_numbers = occupation_numbers[fallback_np.newaxis, :]

    T = transmission_matrix

    output_number_of_modes = T.shape[0]
    input_number_of_modes = T.shape[1]

    input_modes = to_first_quantized(input_occupation)
    number_of_particles = len(input_modes)

    detected_numbers_of_particles = fallback_np.sum(occupation_numbers, axis=1)

    valid_outcomes = detected_numbers_of_particles <= number_of_particles

    probabilities = fallback_np.zeros(len(occupation_numbers))

    if number_of_particles == 0:
        probabilities[detected_numbers_of_particles == 0] = 1.0
        return probabilities

    if not fallback_np.any(valid_outcomes):
        return probabilities

    valid_occupation_numbers = occupation_numbers[valid_outcomes]

    if connector.np.isscalar(particle_overlap):
        G = particle_overlap * np.ones(
            (number_of_particles, number_of_particles),
            dtype=complex_dtype,
        ) + (1.0 - particle_overlap) * np.identity(
            number_of_particles,
            dtype=complex_dtype,
        )

        input_norm = _uniform_input_norm(
            input_occupation=input_occupation,
            particle_overlap=particle_overlap,
            connector=connector,
        )
    else:
        G = particle_overlap

        input_norm = _general_input_norm(
            input_occupation=input_occupation,
            particle_overlap_matrix=particle_overlap,
            connector=connector,
        )

    loss_kernel = (
        np.identity(input_number_of_modes, dtype=complex_dtype) - np.conj(T).T @ T
    )

    loss_kernel = loss_kernel[fallback_np.ix_(input_modes, input_modes)]

    B_loss = G * loss_kernel

    B_detected = []

    for mode in range(output_number_of_modes):
        vector = T[mode, input_modes]

        B_detected.append(G * np.outer(vector, np.conj(vector)))

    loss_subset_row_sums = _precompute_subset_row_sums(B_loss, connector)

    detected_subset_row_sums = np.array(
        [_precompute_subset_row_sums(B_mode, connector) for B_mode in B_detected]
    )

    target = np.max(valid_occupation_numbers, axis=0)

    polynomial_shape = tuple(target + 1)

    coefficients = np.zeros(polynomial_shape, dtype=complex_dtype)

    vacuum_index = (0,) * output_number_of_modes

    polynomial = np.zeros(polynomial_shape, dtype=complex_dtype)
    buffer = np.zeros_like(polynomial)

    for subset in range(1, 1 << number_of_particles):
        sign = -1 if (number_of_particles - subset.bit_count()) % 2 else 1

        polynomial.fill(0.0)
        polynomial[vacuum_index] = 1.0

        constants = loss_subset_row_sums[subset]
        linear_coefficients_by_mode = detected_subset_row_sums[:, subset, :]

        for row in range(number_of_particles):
            multiply_by_linear_truncated(
                polynomial=polynomial,
                constant=constants[row],
                linear_coefficients=linear_coefficients_by_mode[:, row],
                out=buffer,
            )

            polynomial, buffer = buffer, polynomial

        coefficients += sign * polynomial

    coefficients /= input_norm

    valid_probabilities = np.real(coefficients[tuple(valid_occupation_numbers.T)])

    probabilities[valid_outcomes] = valid_probabilities

    return probabilities


def _uniform_input_norm(input_occupation, particle_overlap, connector):
    norm = 1.0 + 0.0j

    for n in input_occupation:
        norm *= sum(
            comb(n, k)
            * particle_overlap**k
            * (1.0 - particle_overlap) ** (n - k)
            * factorial(k)
            for k in range(n + 1)
        )

    return norm


def _general_input_norm(
    input_occupation,
    particle_overlap_matrix,
    connector,
):
    np = connector.np

    norm = 1.0
    start = 0

    indices = np.zeros(len(particle_overlap_matrix), dtype=int)

    for occupation in input_occupation:
        stop = start + occupation

        if occupation > 1:
            indices[start:stop] = 1

            norm *= connector.permanent(particle_overlap_matrix, indices, indices)

            indices[start:stop] = 0

        start = stop

    return norm


def _precompute_subset_row_sums(matrix, connector):
    number_of_particles = matrix.shape[0]
    number_of_subsets = 1 << number_of_particles

    subset_row_sums = connector.np.zeros(
        (number_of_subsets, number_of_particles), dtype=matrix.dtype
    )

    for subset in range(1, number_of_subsets):
        least_significant_bit = subset & -subset
        column = least_significant_bit.bit_length() - 1
        previous_subset = subset ^ least_significant_bit

        subset_row_sums[subset] = subset_row_sums[previous_subset] + matrix[:, column]

    return subset_row_sums
