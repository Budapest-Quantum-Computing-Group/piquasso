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

from math import comb, factorial
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from piquasso._math.combinatorics import partitions


def generate_marginal_samples(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
    shots: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """Generate exact marginal samples on the requested output modes.

    The implementation uses the Laguerre generating-function coefficients for
    collided Fock inputs, restricted to the total-degree basis ``|alpha| <= n``.
    The scaled diagonal coefficients ``alpha! P[alpha, alpha]`` are computed once,
    then samples are drawn mode-by-mode from prefix probabilities.
    """
    initial_state = np.asarray(initial_state, dtype=int)
    modes = tuple(int(mode) for mode in modes)

    number_of_particles = int(np.sum(initial_state))
    number_of_modes = len(modes)

    if number_of_particles == 0:
        return [(0,) * number_of_modes for _ in range(shots)]

    compositions, composition_indices = _get_compositions(
        number_of_modes, number_of_particles
    )
    coefficients = _get_scaled_diagonal_coefficients(
        interferometer=np.asarray(interferometer, dtype=np.complex128),
        initial_state=initial_state,
        modes=modes,
        compositions=compositions,
        composition_indices=composition_indices,
        number_of_particles=number_of_particles,
    )

    return _sample_modes_from_coefficients(
        coefficients=coefficients,
        compositions=compositions,
        number_of_particles=number_of_particles,
        number_of_modes=number_of_modes,
        shots=shots,
        rng=rng,
    )


def marginal_strategy_is_preferred(
    number_of_particles: int,
    number_of_measured_modes: int,
    number_of_occupied_modes: int,
    number_of_modes: int,
    shots: int,
) -> bool:
    """Return whether the marginal sampler is likely cheaper than discarding.

    This deliberately favors the existing full sampler unless the measured subset
    and coefficient table are small. The marginal path's memory is quadratic in
    the number of total-degree compositions.
    """
    n = number_of_particles
    k = number_of_measured_modes

    if not (0 < k < number_of_modes):
        return False

    coefficient_count = comb(n + k, k)

    if coefficient_count > 2500:
        return False

    marginal_cost = max(number_of_occupied_modes, 1) * coefficient_count**2
    full_sampler_cost = shots * max(n, 1) * number_of_modes * 2 ** min(n, 48)

    return marginal_cost < full_sampler_cost


def _get_compositions(
    number_of_modes: int, number_of_particles: int
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    """Return all ``number_of_modes``-tuples with total degree at most ``n``."""
    compositions = partitions(number_of_modes + 1, number_of_particles)[
        :, :number_of_modes
    ].astype(int)
    composition_indices = {
        tuple(composition): index for index, composition in enumerate(compositions)
    }

    return compositions, composition_indices


def _get_scaled_diagonal_coefficients(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
    compositions: np.ndarray,
    composition_indices: Dict[Tuple[int, ...], int],
    number_of_particles: int,
) -> np.ndarray:
    number_of_compositions = len(compositions)
    zero_index = composition_indices[(0,) * len(modes)]

    product = sparse.lil_matrix(
        (number_of_compositions, number_of_compositions),
        dtype=np.complex128,
    )
    product[zero_index, zero_index] = 1.0
    product = product.tocsr()

    occupied_modes = np.nonzero(initial_state)[0]
    occupations = initial_state[occupied_modes]
    modes_array = np.asarray(modes, dtype=int)
    submatrix = interferometer[np.ix_(modes_array, occupied_modes)]
    degrees = np.sum(compositions, axis=1)

    for index, occupation in enumerate(occupations):
        product = _multiply_laguerre_factor(
            product=product,
            column=submatrix[:, index],
            occupation=int(occupation),
            compositions=compositions,
            composition_indices=composition_indices,
            degrees=degrees,
            number_of_particles=number_of_particles,
        )

    alpha_factorials = np.array(
        [
            np.prod([factorial(int(value)) for value in composition])
            for composition in compositions
        ],
        dtype=np.float64,
    )

    return product.diagonal().real * alpha_factorials


def _multiply_laguerre_factor(
    product: sparse.csr_matrix,
    column: np.ndarray,
    occupation: int,
    compositions: np.ndarray,
    composition_indices: Dict[Tuple[int, ...], int],
    degrees: np.ndarray,
    number_of_particles: int,
) -> sparse.csr_matrix:
    number_of_compositions = len(compositions)
    result = sparse.csr_matrix(
        (number_of_compositions, number_of_compositions),
        dtype=np.complex128,
    )

    for degree in range(occupation + 1):
        weights = _get_linear_form_power_coefficients(
            column=column,
            degree=degree,
            compositions=compositions,
            degrees=degrees,
        )
        shift_z = _get_shift_operator(
            weights=weights,
            compositions=compositions,
            composition_indices=composition_indices,
            number_of_particles=number_of_particles,
        )
        shift_w = _get_shift_operator(
            weights=np.conj(weights),
            compositions=compositions,
            composition_indices=composition_indices,
            number_of_particles=number_of_particles,
        )

        result = result + (comb(occupation, degree) / factorial(degree)) * (
            shift_z @ product @ shift_w.T
        )

    return result


def _get_linear_form_power_coefficients(
    column: np.ndarray,
    degree: int,
    compositions: np.ndarray,
    degrees: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(len(compositions), dtype=np.complex128)

    for index in np.where(degrees == degree)[0]:
        composition = compositions[index]
        multinomial = factorial(degree)
        for value in composition:
            multinomial //= factorial(int(value))

        weights[index] = multinomial * np.prod(column**composition)

    return weights


def _get_shift_operator(
    weights: np.ndarray,
    compositions: np.ndarray,
    composition_indices: Dict[Tuple[int, ...], int],
    number_of_particles: int,
) -> sparse.csr_matrix:
    nonzero_indices = np.nonzero(weights)[0]
    number_of_compositions = len(compositions)

    rows: List[int] = []
    columns: List[int] = []
    data: List[complex] = []

    for column_index, base in enumerate(compositions):
        for weight_index in nonzero_indices:
            shifted = base + compositions[weight_index]

            if np.sum(shifted) > number_of_particles:
                continue

            rows.append(composition_indices[tuple(shifted)])
            columns.append(column_index)
            data.append(weights[weight_index])

    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(number_of_compositions, number_of_compositions),
        dtype=np.complex128,
    )


def _sample_modes_from_coefficients(
    coefficients: np.ndarray,
    compositions: np.ndarray,
    number_of_particles: int,
    number_of_modes: int,
    shots: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    groups: List[Tuple[int, Tuple[int, ...]]] = [(shots, ())]

    for _ in range(number_of_modes):
        next_groups: List[Tuple[int, Tuple[int, ...]]] = []

        for count, prefix in groups:
            remaining_particles = number_of_particles - sum(prefix)
            probabilities = np.array(
                [
                    _calculate_prefix_probability(
                        prefix + (value,),
                        coefficients=coefficients,
                        compositions=compositions,
                    )
                    for value in range(remaining_particles + 1)
                ],
                dtype=np.float64,
            )
            probabilities = _normalize_probabilities(probabilities)
            sampled_counts = rng.multinomial(count, probabilities)

            for value, value_count in enumerate(sampled_counts):
                if value_count:
                    next_groups.append((int(value_count), prefix + (value,)))

        groups = next_groups

    samples: List[Tuple[int, ...]] = []
    for count, prefix in groups:
        samples.extend([prefix] * count)

    rng.shuffle(samples)

    return samples


def _calculate_prefix_probability(
    prefix: Tuple[int, ...],
    coefficients: np.ndarray,
    compositions: np.ndarray,
) -> float:
    prefix_array = np.asarray(prefix, dtype=int)
    prefix_length = len(prefix)

    active = compositions[:, :prefix_length]
    trailing = compositions[:, prefix_length:]

    mask = np.all(active >= prefix_array, axis=1) & np.all(trailing == 0, axis=1)
    if not np.any(mask):
        return 0.0

    selected = active[mask]
    signs = (-1.0) ** np.sum(selected - prefix_array, axis=1)
    binomials = np.prod(
        [
            [comb(int(alpha), int(value)) for alpha, value in zip(row, prefix_array)]
            for row in selected
        ],
        axis=1,
    )

    probability = np.sum(coefficients[mask] * binomials * signs)

    return float(probability)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if np.any(probabilities < -1e-9):
        raise ValueError("The marginal sampling probabilities are numerically invalid.")

    probabilities = np.where(probabilities < 1e-12, 0.0, probabilities)
    total = np.sum(probabilities)

    if total <= 0.0:
        raise ValueError("The marginal sampling probabilities are numerically invalid.")

    return probabilities / total
