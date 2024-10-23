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

from typing import Tuple, List

import scipy
import numpy as np

from itertools import repeat
from functools import lru_cache

from scipy.special import factorial

from .state import GaussianState
from .probabilities import (
    calculate_click_probability_nondisplaced,
    calculate_click_probability,
)

from piquasso.instructions import gates

from piquasso.api.result import Result
from piquasso.api.instruction import Instruction
from piquasso.api.exceptions import InvalidInstruction

from piquasso._math.indices import get_operator_index, get_auxiliary_operator_index
from piquasso._math.decompositions import (
    williamson,
    decompose_adjacency_matrix_into_circuit,
)


def passive_linear(
    state: GaussianState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    modes = instruction.modes
    passive_block: np.ndarray = instruction._get_passive_block(
        state._connector, state._config
    )

    _apply_passive_linear(state, passive_block, modes=modes)

    return Result(state=state)


def _apply_passive_linear(state, passive_block, modes):
    connector = state._connector

    state._m = connector.assign(state._m, (modes,), passive_block @ state._m[modes,])

    _apply_passive_linear_to_C_and_G(state, passive_block, modes=modes)


def _apply_passive_linear_to_C_and_G(
    state: GaussianState, T: np.ndarray, modes: Tuple[int, ...]
) -> None:
    connector = state._connector

    index = get_operator_index(modes)

    state._C = connector.assign(
        state._C, index, T.conjugate() @ state._C[index] @ T.transpose()
    )
    state._G = connector.assign(state._G, index, T @ state._G[index] @ T.transpose())

    auxiliary_modes = state._get_auxiliary_modes(modes)

    if len(auxiliary_modes) != 0:
        _apply_passive_linear_to_auxiliary_modes(state, T, modes, auxiliary_modes)


def _apply_passive_linear_to_auxiliary_modes(
    state: GaussianState,
    T: np.ndarray,
    modes: Tuple[int, ...],
    auxiliary_modes: Tuple[int, ...],
) -> None:
    connector = state._connector
    np = connector.np

    auxiliary_index = get_auxiliary_operator_index(modes, auxiliary_modes)

    state._C = connector.assign(
        state._C, auxiliary_index, T.conjugate() @ state._C[auxiliary_index]
    )
    state._G = connector.assign(
        state._G, auxiliary_index, T @ state._G[auxiliary_index]
    )

    assign_index = np.ix_(np.arange(state.d), np.array(modes))
    state._C = connector.assign(
        state._C, assign_index, np.conj(state._C[modes, :]).transpose()
    )
    state._G = connector.assign(state._G, assign_index, state._G[modes, :].transpose())


def linear(
    state: GaussianState, instruction: gates._ActiveLinearGate, shots: int
) -> Result:
    modes = instruction.modes
    passive_block: np.ndarray = instruction._get_passive_block(
        state._connector, state._config
    )
    active_block: np.ndarray = instruction._get_active_block(
        state._connector, state._config
    )

    _apply_linear(state, passive_block, active_block, modes)

    return Result(state=state)


def _apply_linear(state, passive_block, active_block, modes):
    connector = state._connector
    np = connector.np

    passive_part = passive_block @ state._m[(modes,)]

    active_part = active_block @ np.conj(state._m[modes,])

    state._m = connector.assign(state._m, (modes,), passive_part + active_part)

    _apply_linear_to_C_and_G(state, passive_block, active_block, modes)


def _apply_linear_to_C_and_G(
    state: GaussianState, P: np.ndarray, A: np.ndarray, modes: Tuple[int, ...]
) -> None:
    connector = state._connector
    np = connector.np

    index = get_operator_index(modes)

    original_C = state._C[index]
    original_G = state._G[index]

    state._G = connector.assign(
        state._G,
        index,
        P @ original_G @ P.transpose()
        + A @ original_G.conjugate().transpose() @ A.transpose()
        + P @ (original_C.transpose() + np.identity(len(modes))) @ A.transpose()
        + A @ original_C @ P.transpose(),
    )

    state._C = connector.assign(
        state._C,
        index,
        P.conjugate() @ original_C @ P.transpose()
        + A.conjugate()
        @ (original_C.transpose() + np.identity(len(modes)))
        @ A.transpose()
        + P.conjugate() @ original_G.conjugate().transpose() @ A.transpose()
        + A.conjugate() @ original_G @ P.transpose(),
    )

    auxiliary_modes = state._get_auxiliary_modes(modes)

    if len(auxiliary_modes) != 0:
        _apply_linear_to_auxiliary_modes(state, P, A, modes, auxiliary_modes)


def _apply_linear_to_auxiliary_modes(
    state: GaussianState,
    P: np.ndarray,
    A: np.ndarray,
    modes: Tuple[int, ...],
    auxiliary_modes: Tuple[int, ...],
) -> None:
    connector = state._connector
    np = connector.np

    auxiliary_index = get_auxiliary_operator_index(modes, auxiliary_modes)

    auxiliary_C = state._C[auxiliary_index]
    auxiliary_G = state._G[auxiliary_index]

    state._C = connector.assign(
        state._C,
        auxiliary_index,
        P.conjugate() @ auxiliary_C + A.conjugate() @ auxiliary_G,
    )

    state._G = connector.assign(
        state._G, auxiliary_index, P @ auxiliary_G + A @ auxiliary_C
    )

    assign_index = np.ix_(np.arange(state.d), np.array(modes))

    state._C = connector.assign(
        state._C, assign_index, state._C[modes, :].conjugate().transpose()
    )
    state._G = connector.assign(state._G, assign_index, state._G[modes, :].transpose())


def displacement(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    connector = state._connector
    np = connector.np

    modes = instruction.modes
    r = instruction._all_params["r"]
    phi = instruction._all_params["phi"]

    indices = np.ix_(np.array(modes))

    state._m = connector.assign(
        state._m, indices, state._m[indices] + r * np.exp(1j * phi)
    )

    return Result(state=state)


def generaldyne_measurement(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    modes = instruction.modes
    detection_covariance = instruction._all_params["detection_covariance"]

    samples = _get_generaldyne_samples(
        state,
        modes,
        shots,
        detection_covariance,
    )

    # NOTE: We choose the last sample for multiple shots.
    sample = samples[-1]

    d = len(state)

    auxiliary_modes = state._get_auxiliary_modes(modes)
    outer_indices = _map_modes_to_xpxp_indices(auxiliary_modes)

    evolved_state = _get_generaldyne_evolved_state(
        state,
        sample,
        modes,
        detection_covariance,
    )

    evolved_mean = np.zeros(2 * d)
    evolved_mean[outer_indices] = evolved_state.xpxp_mean_vector

    evolved_cov = np.identity(2 * d) * state._config.hbar
    evolved_cov[np.ix_(outer_indices, outer_indices)] = (
        evolved_state.xpxp_covariance_matrix
    )

    state.xpxp_mean_vector = evolved_mean
    state.xpxp_covariance_matrix = evolved_cov

    return Result(state=state, samples=list(map(tuple, list(samples))))


def _get_generaldyne_samples(state, modes, shots, detection_covariance):
    indices = _map_modes_to_xpxp_indices(modes)

    full_detection_covariance = state._config.hbar * scipy.linalg.block_diag(
        *[detection_covariance] * len(modes)
    )

    mean = state.xpxp_mean_vector[indices]

    cov = (
        state.xpxp_covariance_matrix[np.ix_(indices, indices)]
        + full_detection_covariance
    )

    # HACK: We need tol=1e-7 to avoid Numpy warnings at homodyne detection with
    # squeezed detection covariance. Numpy warns
    # 'covariance is not positive-semidefinite.', but it definitely is. In the SVG
    # decomposition (which numpy uses for the multivariate normal distribution)
    # the U^T and V matrices should equal, but our covariance might contain too
    # large values leading to inequality, resulting in warning.
    #
    # We might be better of setting `check_valid='ignore'` and verifying
    # positive-definiteness for ourselves.
    return state._config.rng.multivariate_normal(
        mean=mean,
        cov=cov,
        size=shots,
        tol=1e-7,
    )


def _get_generaldyne_evolved_state(state, sample, modes, detection_covariance):
    full_detection_covariance = state._config.hbar * scipy.linalg.block_diag(
        *[detection_covariance] * len(modes)
    )

    mean = state.xpxp_mean_vector
    cov = state.xpxp_covariance_matrix

    indices = _map_modes_to_xpxp_indices(modes)
    outer_indices = np.delete(np.arange(2 * state.d), indices)

    mean_measured = mean[indices]
    mean_outer = mean[outer_indices]

    cov_measured = cov[np.ix_(indices, indices)]
    cov_outer = cov[np.ix_(outer_indices, outer_indices)]
    cov_correlation = cov[np.ix_(outer_indices, indices)]

    evolved_cov_outer = (
        cov_outer
        - cov_correlation
        @ np.linalg.inv(cov_measured + full_detection_covariance)
        @ cov_correlation.transpose()
    )

    evolved_r_A = mean_outer + cov_correlation @ np.linalg.inv(
        cov_measured + full_detection_covariance
    ) @ (sample - mean_measured)

    new_state = GaussianState(
        d=len(evolved_r_A) // 2, connector=state._connector, config=state._config
    )

    new_state.xpxp_covariance_matrix = evolved_cov_outer
    new_state.xpxp_mean_vector = evolved_r_A

    return new_state


def _map_modes_to_xpxp_indices(modes):
    indices = []

    for mode in modes:
        indices.extend([2 * mode, 2 * mode + 1])

    return indices


def particle_number_measurement(
    state: GaussianState,
    instruction: Instruction,
    shots: int,
) -> Result:
    samples = _get_particle_number_measurement_samples(state, instruction, shots)

    return Result(state=state, samples=samples)


def _get_particle_number_measurement_samples(
    state: GaussianState,
    instruction: Instruction,
    shots: int,
) -> np.ndarray:
    modes: Tuple[int, ...] = instruction.modes
    connector = state._connector
    config = state._config

    reduced_state = state.reduced(modes)

    d = reduced_state.d

    hbar = config.hbar
    hbar_in_calculations = 2.0

    normalized_cov = (
        hbar / (2.0 * hbar_in_calculations) * reduced_state.xxpp_covariance_matrix
    )
    normalized_mean = (
        np.sqrt(hbar / hbar_in_calculations) * reduced_state.xxpp_mean_vector
    )
    S, D = williamson(normalized_cov, connector)

    T = S @ S.T
    mixed_diag = D - 2 * np.identity(len(D)) / 2
    mixed_diag[mixed_diag < 0.0] = 0.0

    I = np.identity(d)
    F = np.block([[I, 1j * I], [I, -1j * I]]) / np.sqrt(2)
    Q = (F @ T @ F.conj().T + np.identity(2 * d)) / 2
    B = -np.linalg.inv(Q)[d:, :d]

    sqrt_cov_1 = S @ np.sqrt(mixed_diag)
    sqrt_cov_2 = np.linalg.cholesky(T + np.identity(2 * d))

    samples = np.empty(shape=(shots, d), dtype=int)

    for idx in range(shots):
        sample = _generate_sample(
            B,
            sqrt_cov_1,
            sqrt_cov_2,
            mean=normalized_mean,
            connector=reduced_state._connector,
            config=config,
        )

        samples[idx] = sample

    return samples


def _generate_sample(B, sqrt_cov_1, sqrt_cov_2, mean, connector, config):
    d = len(B)
    cutoff = config.measurement_cutoff
    rng = config.rng

    possible_choices = np.arange(cutoff)

    sample = np.zeros(d, dtype=int)

    pure_mean = mean + sqrt_cov_1 @ rng.normal(size=2 * d)
    pure_mean_complex = (pure_mean[:d] + 1j * pure_mean[d:]) / 2

    evolved_mean = pure_mean + sqrt_cov_2 @ rng.normal(size=2 * d)
    evolved_mean_complex = (evolved_mean[:d] + 1j * evolved_mean[d:]) / 2
    gamma = pure_mean_complex.conj() + B @ (evolved_mean_complex - pure_mean_complex)

    for mode in range(d):
        gamma -= evolved_mean_complex[mode] * B[:, mode]
        mode_p_1 = mode + 1
        matrix = B[:mode_p_1, :mode_p_1]
        diagonal = gamma[:mode_p_1]
        partial_sample = sample[:mode_p_1]
        lhaf_values = connector.loop_hafnian_batch(
            matrix, diagonal, partial_sample, cutoff
        )
        weights = np.abs(lhaf_values) ** 2 / factorial(possible_choices)
        weights /= np.sum(weights)

        sample[mode] = rng.choice(possible_choices, p=weights)

    return sample


def threshold_measurement(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    """
    NOTE: The same logic is used here, as for the particle number measurement.
    However, at threshold measurement there is no sense of measurement cutoff,
    therefore it is set to 2 to make the logic sensible in this case as well.

    Also note, that one could speed up this calculation by not calculating the
    probability of clicks (i.e. 1 as sample), and argue that the probability of a
    click is equal to one minus the probability of no click.
    """
    if state._config.use_torontonian:
        samples = _generate_threshold_samples_using_torontonian(
            state, instruction, shots
        )
    else:
        samples = _generate_threshold_samples_using_hafnian(state, instruction, shots)

    return Result(state=state, samples=samples)


def _generate_threshold_samples_using_torontonian(state, instruction, shots):
    is_displaced = state._is_displaced()
    rng = state._config.rng
    hbar = state._config.hbar

    modes = instruction.modes

    @lru_cache(state._config.cache_size)
    def get_probability(
        subspace_modes: Tuple[int, ...], occupation_numbers: Tuple[int, ...]
    ) -> float:
        reduced_state = state.reduced(subspace_modes)

        if not is_displaced:
            return calculate_click_probability_nondisplaced(
                reduced_state.xpxp_covariance_matrix / hbar,
                occupation_numbers,
            )

        return calculate_click_probability(
            reduced_state.xpxp_covariance_matrix / hbar,
            reduced_state.xpxp_mean_vector / np.sqrt(hbar),
            occupation_numbers,
        )

    samples = []

    for _ in repeat(None, shots):
        sample: List[int] = []

        previous_probability = 1.0

        for mode_index in range(1, len(modes) + 1):
            subspace_modes = tuple(modes[:mode_index])

            occupation_numbers = tuple(sample + [0])

            probability = get_probability(
                subspace_modes=subspace_modes, occupation_numbers=occupation_numbers
            )
            conditional_probability = probability / previous_probability

            choice: int
            guess = rng.uniform()

            if guess < conditional_probability:
                choice = 0
                previous_probability *= conditional_probability
            else:
                choice = 1
                previous_probability *= 1 - conditional_probability

            sample.append(choice)

        samples.append(tuple(sample))

    return samples


def _generate_threshold_samples_using_hafnian(state, instruction, shots):
    samples = _get_particle_number_measurement_samples(state, instruction, shots)

    threshold_samples = []

    for sample in samples:
        threshold_samples.append(
            [1 if photon_number else 0 for photon_number in sample]
        )

    return threshold_samples


def homodyne_measurement(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    phi = instruction._all_params["phi"]

    modes = instruction.modes

    phaseshift = np.identity(len(instruction.modes)) * np.exp(-1j * phi)

    _apply_passive_linear(state, phaseshift, modes=modes)

    result = generaldyne_measurement(state, instruction, shots)

    return Result(state=state, samples=result.samples)


def vacuum(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def mean(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    state.xpxp_mean_vector = instruction._all_params["mean"] * np.sqrt(
        state._config.hbar
    )

    return Result(state=state)


def covariance(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    state.xpxp_covariance_matrix = instruction._all_params["cov"] * state._config.hbar

    return Result(state=state)


def graph(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    """
    TODO: Find a better solution for multiple operations.
    """
    squeezings, interferometer = decompose_adjacency_matrix_into_circuit(
        adjacency_matrix=instruction._params["adjacency_matrix"],
        mean_photon_number=instruction._params["mean_photon_number"],
        connector=state._connector,
    )

    for mode, r in zip(instruction.modes, squeezings):
        _apply_linear(
            state=state,
            passive_block=np.array([[np.cosh(r)]]),
            active_block=np.array([[np.sinh(r)]]),
            modes=(mode,),
        )

    _apply_passive_linear(state, interferometer, instruction.modes)

    return Result(state=state)


def deterministic_gaussian_channel(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    X = instruction._all_params["X"]
    Y = instruction._all_params["Y"] * state._config.hbar
    modes = instruction.modes

    if state._config.validate and len(X) != 2 * len(modes) or len(Y) != 2 * len(modes):
        raise InvalidInstruction(
            f"The instruction should be specified for '{len(modes)}' modes: "
            f"instruction={instruction}"
        )

    indices = _map_modes_to_xpxp_indices(modes)
    matrix_indices = np.ix_(indices, indices)

    mean_vector = state.xpxp_mean_vector
    covariance_matrix = state.xpxp_covariance_matrix

    embedded_X = np.identity(len(mean_vector), dtype=state._config.complex_dtype)
    embedded_X[matrix_indices] = X

    embedded_Y = np.zeros_like(embedded_X)
    embedded_Y[matrix_indices] = Y

    state.xpxp_mean_vector = embedded_X @ mean_vector
    state.xpxp_covariance_matrix = (
        embedded_X @ covariance_matrix @ embedded_X.T + embedded_Y
    )

    return Result(state=state)
