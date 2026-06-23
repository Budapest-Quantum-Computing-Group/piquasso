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

from .utils import calculate_inner_product, calculate_lossy_density_matrix_element


def get_ideal_particle_number_probability(
    occupation_number,
    postselections,
    interferometer,
    occupation_numbers,
    coefficients,
    connector,
):
    np = connector.np
    fallback_np = connector.fallback_np

    output_number_of_particles = fallback_np.sum(occupation_number) + sum(
        postselections.values()
    )

    sum_ = 0.0

    total_number_of_modes = len(interferometer)

    postselected_modes = tuple(postselections.keys())
    active_modes = tuple(
        [i for i in range(total_number_of_modes) if i not in postselected_modes]
    )
    postselected_photons = tuple(postselections.values())

    full_occupation_number = fallback_np.zeros(total_number_of_modes, dtype=int)

    full_occupation_number[active_modes,] = occupation_number
    full_occupation_number[postselected_modes,] = postselected_photons

    for index, input_occupation_number in enumerate(occupation_numbers):
        if output_number_of_particles != fallback_np.sum(input_occupation_number):
            continue

        inner_product = calculate_inner_product(
            interferometer=interferometer,
            input=input_occupation_number,
            output=full_occupation_number,
            connector=connector,
        )
        coefficient = coefficients[index]

        sum_ += coefficient * inner_product

    return np.abs(sum_) ** 2


def get_lossy_particle_number_probability(
    occupation_number,
    postselections,
    transmission_matrix,
    occupation_numbers,
    coefficients,
    connector,
):
    np = connector.np
    fallback_np = connector.fallback_np

    total_number_of_modes = len(transmission_matrix)

    postselected_modes = tuple(postselections.keys())
    active_modes = tuple(
        mode for mode in range(total_number_of_modes) if mode not in postselected_modes
    )
    postselected_photons = tuple(postselections.values())

    full_occupation_number = fallback_np.zeros(
        total_number_of_modes,
        dtype=int,
    )

    full_occupation_number[active_modes,] = occupation_number
    full_occupation_number[postselected_modes,] = postselected_photons

    output_number_of_particles = int(fallback_np.sum(full_occupation_number))

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
                output_left=full_occupation_number,
                output_right=full_occupation_number,
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
