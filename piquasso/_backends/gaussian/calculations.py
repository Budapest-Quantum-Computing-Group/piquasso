#
# Copyright 2021 Budapest Quantum Computing Group
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

import itertools
import scipy
import random
import numpy as np

from itertools import repeat
from functools import lru_cache

from scipy.special import factorial

from .state import GaussianState
from .probabilities import calculate_click_probability

from piquasso.api.result import Result
from piquasso.api.instruction import Instruction
from piquasso.api.errors import InvalidInstruction

from piquasso._math.linalg import reduce_
from piquasso._math.indices import get_operator_index, get_auxiliary_operator_index
from piquasso._math.decompositions import decompose_to_pure_and_mixed


def passive_linear(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    modes = instruction.modes
    T: np.ndarray = instruction._all_params["passive_block"]

    state._m[(modes,)] = (
        T
        @ state._m[
            modes,
        ]
    )

    _apply_passive_linear_to_C_and_G(state, T, modes=modes)

    return Result(state=state)


def _apply_passive_linear_to_C_and_G(
    state: GaussianState, T: np.ndarray, modes: Tuple[int, ...]
) -> None:
    index = get_operator_index(modes)

    state._C[index] = T.conjugate() @ state._C[index] @ T.transpose()
    state._G[index] = T @ state._G[index] @ T.transpose()

    auxiliary_modes = state._get_auxiliary_modes(modes)

    if len(auxiliary_modes) != 0:
        _apply_passive_linear_to_auxiliary_modes(state, T, modes, auxiliary_modes)


def _apply_passive_linear_to_auxiliary_modes(
    state: GaussianState,
    T: np.ndarray,
    modes: Tuple[int, ...],
    auxiliary_modes: Tuple[int, ...],
) -> None:
    auxiliary_index = get_auxiliary_operator_index(modes, auxiliary_modes)

    state._C[auxiliary_index] = T.conjugate() @ state._C[auxiliary_index]
    state._G[auxiliary_index] = T @ state._G[auxiliary_index]

    state._C[:, modes] = np.conj(state._C[modes, :]).transpose()
    state._G[:, modes] = state._G[modes, :].transpose()


def linear(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    modes = instruction.modes
    passive_block: np.ndarray = instruction._all_params["passive_block"]
    active_block: np.ndarray = instruction._all_params["active_block"]

    state._m[(modes,)] = passive_block @ state._m[(modes,)] + active_block @ np.conj(
        state._m[
            modes,
        ]
    )

    _apply_linear_to_C_and_G(state, passive_block, active_block, modes)

    return Result(state=state)


def _apply_linear_to_C_and_G(
    state: GaussianState, P: np.ndarray, A: np.ndarray, modes: Tuple[int, ...]
) -> None:
    index = get_operator_index(modes)

    original_C = state._C[index]
    original_G = state._G[index]

    state._G[index] = (
        P @ original_G @ P.transpose()
        + A @ original_G.conjugate().transpose() @ A.transpose()
        + P @ (original_C.transpose() + np.identity(len(modes))) @ A.transpose()
        + A @ original_C @ P.transpose()
    )

    state._C[index] = (
        P.conjugate() @ original_C @ P.transpose()
        + A.conjugate()
        @ (original_C.transpose() + np.identity(len(modes)))
        @ A.transpose()
        + P.conjugate() @ original_G.conjugate().transpose() @ A.transpose()
        + A.conjugate() @ original_G @ P.transpose()
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
    auxiliary_index = get_auxiliary_operator_index(modes, auxiliary_modes)

    auxiliary_C = state._C[auxiliary_index]
    auxiliary_G = state._G[auxiliary_index]

    state._C[auxiliary_index] = (
        P.conjugate() @ auxiliary_C + A.conjugate() @ auxiliary_G
    )

    state._G[auxiliary_index] = P @ auxiliary_G + A @ auxiliary_C

    state._C[:, modes] = state._C[modes, :].conjugate().transpose()
    state._G[:, modes] = state._G[modes, :].transpose()


def displacement(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    modes = instruction.modes
    displacement_vector: np.ndarray = instruction._all_params["displacement_vector"]

    state._m[
        modes,
    ] += displacement_vector

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
    evolved_cov[
        np.ix_(outer_indices, outer_indices)
    ] = evolved_state.xpxp_covariance_matrix

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

    state = GaussianState(d=len(evolved_r_A) // 2, config=state._config)

    state.xpxp_covariance_matrix = evolved_cov_outer
    state.xpxp_mean_vector = evolved_r_A

    return state


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
) -> List[Tuple[int, ...]]:

    modes: Tuple[int, ...] = instruction.modes

    reduced_state = state.reduced(modes)

    reduced_modes = tuple(range(len(modes)))

    pure_covariance, mixed_contribution = decompose_to_pure_and_mixed(
        reduced_state.xxpp_covariance_matrix,
        hbar=state._config.hbar,
    )
    pure_state = GaussianState(len(reduced_state), config=state._config)
    pure_state.xxpp_covariance_matrix = pure_covariance

    heterodyne_detection_covariance = np.identity(2)

    samples = []

    for _ in itertools.repeat(None, shots):
        pure_state.xxpp_mean_vector = state._config.rng.multivariate_normal(
            reduced_state.xxpp_mean_vector, mixed_contribution
        )

        heterodyne_sample = _get_generaldyne_samples(
            state=pure_state,
            modes=reduced_modes,
            shots=1,
            detection_covariance=heterodyne_detection_covariance,
        )[0]

        sample: Tuple[int, ...] = tuple()

        heterodyne_measured_modes = reduced_modes

        for _ in itertools.repeat(None, len(reduced_modes)):
            heterodyne_sample = heterodyne_sample[2:]
            heterodyne_measured_modes = heterodyne_measured_modes[1:]

            evolved_state = _get_generaldyne_evolved_state(
                pure_state,
                heterodyne_sample,
                heterodyne_measured_modes,
                heterodyne_detection_covariance,
            )

            choice = _get_particle_number_choice(
                evolved_state,
                sample,
            )

            sample = sample + (choice,)

        samples.append(sample)

    return samples


def _get_particle_number_choice(
    state: GaussianState,
    previous_sample: Tuple[int, ...],
) -> int:
    r"""
    The original equations are

    .. math::
        A = X - X Q^{-1} = \begin{bmatrix}
            B & 0 \\
            0 & B^* \\
        \end{bmatrix}

    with

    .. math::
        X = \begin{bmatrix}
            0 & \mathbb{I} \\
            \mathbb{I} & 0 \\
        \end{bmatrix}

    But then one could write

    .. math::
        B = - Q^{-1}[d:, :d]

    Everything can be rewritten using :math:`B` in the following manner:

    ..math::
        \gamma = \alpha^* - \alpha B \\
        p_{vac} = \exp
            \left (
                - \Re(\gamma \alpha)
            \right )
            \sqrt{
                \operatorname{det}(\mathbb{I} - B^* B)
            } \\
        p_{S} = p_{vac} \frac{
            | \operatorname{lhaf}(\operatorname{filldiag}(B_S, \gamma_S))
        }{
            S!
        }
    """

    d = len(state)

    B = -np.linalg.inv(state.Q_matrix)[d:, :d]
    alpha = state.complex_displacement[:d]

    gamma = alpha.conj() - alpha @ B

    weights: np.ndarray = np.array([])

    possible_choices = tuple(range(state._config.measurement_cutoff))

    for n in possible_choices:
        occupation_numbers = previous_sample + (n,)

        B_reduced = reduce_(B, reduce_on=occupation_numbers)
        gamma_reduced = reduce_(gamma, reduce_on=occupation_numbers)

        np.fill_diagonal(B_reduced, gamma_reduced)

        weight = abs(state._config.loop_hafnian_function(B_reduced)) ** 2 / factorial(n)
        weights = np.append(weights, weight)

    weights /= np.sum(weights)

    return np.random.choice(possible_choices, p=weights)


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


def _generate_threshold_samples_using_torontonian(
    state,
    instruction,
    shots,
):
    if not np.allclose(state.xpxp_mean_vector, np.zeros_like(state.xpxp_mean_vector)):
        raise NotImplementedError(
            "Threshold measurement for displaced states are not supported: "
            f"xpxp_mean_vector={state.xpxp_mean_vector}"
        )

    modes = instruction.modes

    @lru_cache(state._config.cache_size)
    def get_probability(
        subspace_modes: Tuple[int, ...], occupation_numbers: Tuple[int, ...]
    ) -> float:
        reduced_state = state.reduced(subspace_modes)

        return calculate_click_probability(
            reduced_state.xxpp_covariance_matrix,
            occupation_numbers,
            state._config.hbar,
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
            guess = random.uniform(0.0, 1.0)

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

    instruction_copy: Instruction = instruction.copy()  # type: ignore

    phaseshift = np.identity(len(instruction.modes)) * np.exp(-1j * phi)
    instruction_copy._extra_params["passive_block"] = phaseshift

    result = passive_linear(state, instruction_copy, shots)

    result = generaldyne_measurement(result.state, instruction, shots)

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
    instruction._all_params["squeezing"].modes = instruction.modes
    instruction._all_params["interferometer"].modes = instruction.modes

    result = linear(state, instruction._all_params["squeezing"], shots)

    return passive_linear(
        result.state, instruction._all_params["interferometer"], shots
    )


def deterministic_gaussian_channel(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    X = instruction._all_params["X"]
    Y = instruction._all_params["Y"] * state._config.hbar
    modes = instruction.modes

    if len(X) != 2 * len(modes) or len(Y) != 2 * len(modes):
        raise InvalidInstruction(
            f"The instruction should be specified for '{len(modes)}' modes: "
            f"instruction={instruction}"
        )

    indices = _map_modes_to_xpxp_indices(modes)
    matrix_indices = np.ix_(indices, indices)

    mean_vector = state.xpxp_mean_vector
    covariance_matrix = state.xpxp_covariance_matrix

    embedded_X = np.identity(len(mean_vector), dtype=complex)
    embedded_X[matrix_indices] = X

    embedded_Y = np.zeros_like(embedded_X)
    embedded_Y[matrix_indices] = Y

    state.xpxp_mean_vector = embedded_X @ mean_vector
    state.xpxp_covariance_matrix = (
        embedded_X @ covariance_matrix @ embedded_X.T + embedded_Y
    )

    return Result(state=state)
