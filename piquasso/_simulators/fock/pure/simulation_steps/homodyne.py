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

from typing import List, Optional

from fractions import Fraction
from functools import lru_cache

import numba as nb
import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import factorial, hermite

from piquasso.api.branch import Branch
from piquasso.instructions.measurements import HomodyneMeasurement
from piquasso.api.exceptions import InvalidParameter

from piquasso._math.fock import get_fock_space_basis

from ..state import PureFockState

hermite = lru_cache(maxsize=None)(hermite)


def homodyne_measurement(
    state: PureFockState, instruction: HomodyneMeasurement, shots: int
) -> List[Branch]:
    """Sample homodyne outcomes and return the conditional pure states.

    The measured modes are processed sequentially. For every step we:

    1. regroup the current pure-state amplitudes according to the occupation of the
       mode being measured,
    2. form only the one-mode reduced density matrix,
    3. sample the requested rotated quadrature,
    4. immediately project the measured mode out of the pure state.
    """
    modes = tuple(instruction.modes)

    config = state._config
    cutoff = config.cutoff
    real_dtype = config.dtype
    complex_dtype = config.complex_dtype

    phi = _get_phi_array(
        instruction.params["phi"], number_of_modes=len(modes), real_dtype=real_dtype
    )

    hbar = config.hbar
    sqrt_hbar = np.sqrt(hbar)

    rng = config.rng
    number_of_measured_modes = len(modes)

    uniforms: np.ndarray = np.asarray(
        rng.uniform(size=(shots, number_of_measured_modes)),
        dtype=real_dtype,
    )
    samples: np.ndarray = np.empty(
        shape=(shots, number_of_measured_modes),
        dtype=real_dtype,
    )

    B_matrix = get_B_matrix(cutoff, real_dtype)

    # The first marginal is identical for every shot. Compute it once and draw all
    # first-mode samples together before the branches become outcome-dependent.
    first_mode = modes[0]
    first_projection_data = _get_mode_projection_data(
        d=state.d, cutoff=cutoff, mode=first_mode
    )
    first_density_matrix = _get_one_mode_density_matrix(
        state,
        projection_data=first_projection_data,
        complex_dtype=complex_dtype,
    )
    first_mean_position = _get_dimensionless_mean_position(
        first_density_matrix,
        phi=phi[0],
        real_dtype=real_dtype,
    )

    _homodyne_measurement_one_mode(
        density_matrix=first_density_matrix,
        shots=shots,
        mean_position=first_mean_position,
        B_matrix=B_matrix,
        samples=samples[:, 0],
        uniforms=uniforms[:, 0],
        phi=phi[0],
    )

    branches = []

    for shot_index in range(shots):
        current_state = _project_state_after_one_mode_homodyne(
            state=state,
            q=samples[shot_index, 0],
            phi=phi[0],
            projection_data=first_projection_data,
        )

        # Track how the original mode labels map to the progressively smaller state.
        remaining_original_modes = list(range(state.d))
        del remaining_original_modes[first_mode]

        for measurement_index in range(1, number_of_measured_modes):
            assert current_state is not None

            original_mode = modes[measurement_index]
            current_mode = remaining_original_modes.index(original_mode)
            current_phi = phi[measurement_index]

            projection_data = _get_mode_projection_data(
                d=current_state.d, cutoff=cutoff, mode=current_mode
            )
            density_matrix = _get_one_mode_density_matrix(
                current_state,
                projection_data=projection_data,
                complex_dtype=complex_dtype,
            )
            mean_position = _get_dimensionless_mean_position(
                density_matrix,
                phi=current_phi,
                real_dtype=real_dtype,
            )

            _homodyne_measurement_one_mode(
                density_matrix=density_matrix,
                shots=1,
                mean_position=mean_position,
                B_matrix=B_matrix,
                samples=samples[shot_index, measurement_index : measurement_index + 1],
                uniforms=uniforms[
                    shot_index, measurement_index : measurement_index + 1
                ],
                phi=current_phi,
            )

            current_state = _project_state_after_one_mode_homodyne(
                state=current_state,
                q=samples[shot_index, measurement_index],
                phi=current_phi,
                projection_data=projection_data,
            )

            del remaining_original_modes[current_mode]

        branches.append(
            Branch(
                state=current_state,
                outcome=tuple(
                    (sqrt_hbar * samples[shot_index]).astype(real_dtype, copy=False)
                ),
                frequency=Fraction(1, shots),
            )
        )

    return branches


def _get_phi_array(
    phi: npt.ArrayLike,
    number_of_modes: int,
    real_dtype: type,
) -> np.ndarray:
    """Return one homodyne angle for each measured mode."""
    dtype = np.dtype(real_dtype)
    phi = np.asarray(phi, dtype=dtype)

    if phi.ndim == 0:
        return np.full(
            number_of_modes,
            phi.item(),
            dtype=dtype,
        )

    if phi.ndim != 1 or len(phi) != number_of_modes:
        raise InvalidParameter(
            "'phi' must be either a scalar or a one-dimensional array with "
            f"len(phi) == len(modes). Got phi.ndim={phi.ndim} and len(phi)={len(phi)}."
        )

    return phi


@lru_cache(maxsize=None)
def _get_mode_projection_data(d: int, cutoff: int, mode: int) -> np.ndarray:
    """Return a lookup table for splitting one mode from the Fock basis.

    ``basis_indices[alpha, n]`` is the index in the current d-mode state vector of
    the Fock state whose selected-mode occupation is ``n`` and whose occupations on
    all remaining modes correspond to residual basis index ``alpha``. Missing states
    imposed by the global cutoff are stored as ``-1``.

    The table depends only on ``(d, cutoff, mode)`` and is therefore cached and reused
    across all shots.
    """
    full_basis = np.asarray(
        get_fock_space_basis(d=d, cutoff=cutoff),
        dtype=np.int64,
    )

    measured_occupations = full_basis[:, mode]

    output_indices: np.ndarray[
        tuple[int, ...],
        np.dtype[np.int64],
    ]

    if d == 1:
        output_indices = np.zeros(len(full_basis), dtype=np.int64)
        output_size = 1
    else:
        remaining_modes = tuple(index for index in range(d) if index != mode)
        remaining_occupations = full_basis[:, remaining_modes]

        remaining_basis = np.asarray(
            get_fock_space_basis(d=d - 1, cutoff=cutoff),
            dtype=np.int64,
        )

        output_lookup = {
            tuple(occupation): index for index, occupation in enumerate(remaining_basis)
        }

        output_indices = np.fromiter(
            (output_lookup[tuple(occupation)] for occupation in remaining_occupations),
            dtype=np.int64,
            count=len(full_basis),
        )
        output_size = len(remaining_basis)

    basis_indices = np.full(
        shape=(output_size, cutoff),
        fill_value=-1,
        dtype=np.int64,
    )
    basis_indices[output_indices, measured_occupations] = np.arange(
        len(full_basis), dtype=np.int64
    )

    return basis_indices


@nb.njit(cache=True)
def _calculate_one_mode_density_matrix(state_vector, basis_indices, cutoff):
    density_matrix = np.zeros(
        shape=(cutoff, cutoff),
        dtype=state_vector.dtype,
    )

    for residual_index in range(basis_indices.shape[0]):
        for row_occupation in range(cutoff):
            row_index = basis_indices[residual_index, row_occupation]

            if row_index < 0:
                continue

            row_amplitude = state_vector[row_index]

            for col_occupation in range(cutoff):
                col_index = basis_indices[residual_index, col_occupation]

                if col_index < 0:
                    continue

                density_matrix[
                    row_occupation, col_occupation
                ] += row_amplitude * np.conj(state_vector[col_index])

    return density_matrix


def _get_one_mode_density_matrix(
    state: PureFockState,
    projection_data: np.ndarray,
    complex_dtype: type,
) -> np.ndarray:
    """Return the selected mode's normalized reduced density matrix."""
    dtype = np.dtype(complex_dtype)

    density_matrix = _calculate_one_mode_density_matrix(
        np.asarray(state.state_vector, dtype=dtype),
        projection_data,
        state._config.cutoff,
    )

    trace = np.real(np.trace(density_matrix))

    if np.isclose(trace, 0.0):
        raise RuntimeError("Cannot sample homodyne measurement from a zero state.")

    density_matrix /= trace

    return density_matrix


def _get_dimensionless_mean_position(
    density_matrix: np.ndarray,
    phi: float,
    real_dtype: type,
) -> float:
    r"""Return <Q_phi> / sqrt(hbar) for a one-mode density matrix."""
    dtype = np.dtype(real_dtype)

    photon_numbers: np.ndarray = np.arange(
        1,
        density_matrix.shape[0],
        dtype=dtype,
    )

    mean_annihilation = np.sum(
        np.sqrt(photon_numbers)
        * density_matrix[
            np.arange(1, density_matrix.shape[0]),
            np.arange(density_matrix.shape[0] - 1),
        ]
    )

    phi = dtype.type(phi)
    rotated_real_part = np.real(mean_annihilation) * np.cos(phi) + np.imag(
        mean_annihilation
    ) * np.sin(phi)

    return dtype.type(np.sqrt(dtype.type(2.0)) * rotated_real_part)


@nb.njit(cache=True)
def _project_state_vector(state_vector, basis_indices, overlaps):
    projected_state_vector = np.zeros(
        basis_indices.shape[0],
        dtype=state_vector.dtype,
    )

    for residual_index in range(basis_indices.shape[0]):
        amplitude = projected_state_vector[residual_index]

        for photon_number in range(basis_indices.shape[1]):
            state_index = basis_indices[residual_index, photon_number]

            if state_index >= 0:
                amplitude += state_vector[state_index] * overlaps[photon_number]

        projected_state_vector[residual_index] = amplitude

    return projected_state_vector


def _project_state_after_one_mode_homodyne(
    state: PureFockState,
    q: float,
    phi: float,
    projection_data: np.ndarray,
) -> Optional[PureFockState]:
    cutoff = state._config.cutoff
    real_dtype = np.dtype(state._config.dtype)
    complex_dtype = np.dtype(state._config.complex_dtype)

    overlaps = _get_homodyne_fock_overlaps(
        q=q,
        cutoff=cutoff,
        real_dtype=state._config.dtype,
    ).astype(complex_dtype)

    photon_numbers = np.arange(cutoff, dtype=real_dtype)
    angles = real_dtype.type(phi) * photon_numbers

    phase_factors = np.empty(cutoff, dtype=complex_dtype)
    phase_factors.real = np.cos(angles)
    phase_factors.imag = -np.sin(angles)

    overlaps *= phase_factors

    projected_state_vector = _project_state_vector(
        np.asarray(state.state_vector, dtype=complex_dtype),
        projection_data,
        overlaps,
    )

    norm = np.linalg.norm(projected_state_vector)

    if np.isclose(norm, 0.0):
        raise RuntimeError("Homodyne projection produced a numerically zero state.")

    projected_state_vector /= norm

    if state.d == 1:
        return None

    post_measurement_state = PureFockState(
        d=state.d - 1,
        connector=state._connector,
        config=state._config,
    )

    post_measurement_state.state_vector = projected_state_vector.astype(
        complex_dtype, copy=False
    )

    return post_measurement_state


def _get_homodyne_fock_overlaps(
    q: float,
    cutoff: int,
    real_dtype: type,
) -> np.ndarray:
    dtype = np.dtype(real_dtype)
    q = dtype.type(q)

    overlaps: np.ndarray = np.empty(cutoff, dtype=dtype)

    pi = dtype.type(np.pi)
    half = dtype.type(0.5)
    two = dtype.type(2.0)

    overlaps[0] = pi ** dtype.type(-0.25) * np.exp(-half * q * q)

    if cutoff == 1:
        return overlaps

    overlaps[1] = np.sqrt(two) * q * overlaps[0]

    for n in range(1, cutoff - 1):
        n_real = dtype.type(n)
        n_plus_one = dtype.type(n + 1)

        overlaps[n + 1] = (
            np.sqrt(two / n_plus_one) * q * overlaps[n]
            - np.sqrt(n_real / n_plus_one) * overlaps[n - 1]
        )

    return overlaps


def _homodyne_measurement_one_mode(
    density_matrix,
    shots,
    mean_position,
    B_matrix,
    samples,
    uniforms,
    phi,
):
    poly = get_integral_poly(density_matrix, B_matrix, phi)

    lower, upper, almost_0, almost_1 = get_interval(mean_position, poly)

    for idx in range(shots):
        inverse_sample = uniforms[idx] * (almost_1 - almost_0) + almost_0

        sample = brentq(
            integral_m_inverse_sample,
            a=lower,
            b=upper,
            args=(poly, inverse_sample),
        )

        samples[idx] = sample


@nb.njit(cache=True)
def polyeval(p, x):
    y = p[0] * 0

    for pv in p:
        y = y * x + pv

    return y


@nb.njit(cache=True)
def eval_poly_term(p, x):
    y = polyeval(p, x)

    return y * np.exp(-(x**2))


@nb.njit(cache=True)
def erf(x):
    """Rational approximation (7.1.25) by Abramowitz and Stegun, see
    https://personal.math.ubc.ca/%7Ecbm/aands/page_299.htm
    """
    if x < 0.0:
        return -erf(-x)

    a3 = 0.7478556
    a2 = -0.0958798
    a1 = 0.3480242
    p = 0.47047

    t = 1 / (1 + p * x)

    y = 1 - (a1 * t + a2 * t**2 + a3 * t**3) * np.exp(-(x**2))

    return y


@nb.njit(cache=True)
def integral(poly, x):
    return (erf(x) + 1) / 2 - eval_poly_term(poly, x)


@nb.njit(cache=True)
def integral_m_inverse_sample(x, poly, inverse_sample):
    return integral(poly, x) - inverse_sample


@nb.njit(cache=True)
def add_poly(poly1, poly2):
    n1 = poly1.shape[0]
    n2 = poly2.shape[0]
    n = max(n1, n2)

    new_poly = np.zeros(n, dtype=poly1.dtype)

    for idx in range(n1):
        addend = poly1[n1 - idx - 1]
        new_poly[n - idx - 1] += addend

    for idx in range(n2):
        addend = poly2[n2 - idx - 1]
        new_poly[n - idx - 1] += addend

    return new_poly


@nb.njit(cache=True)
def get_integral_poly(density_matrix, B_matrix, phi):
    size = density_matrix.shape[0]

    integral_poly = np.zeros(1, dtype=B_matrix.dtype)

    starting_index = 0

    for row_idx in range(size):
        for col_idx in range(row_idx):
            stopping_index = row_idx + col_idx
            sliced_I = B_matrix[starting_index : starting_index + stopping_index]

            angle = (row_idx - col_idx) * phi
            matrix_element = density_matrix[row_idx, col_idx]

            rotated_real_part = np.real(matrix_element) * np.cos(angle) + np.imag(
                matrix_element
            ) * np.sin(angle)
            coefficient = rotated_real_part + rotated_real_part

            poly_to_add = sliced_I * coefficient
            starting_index += stopping_index

            integral_poly = add_poly(poly_to_add, integral_poly)

        stopping_index = max(2 * row_idx, 1)
        sliced_I = B_matrix[starting_index : starting_index + stopping_index]
        integral_poly = add_poly(
            sliced_I * np.real(density_matrix[row_idx, row_idx]),
            integral_poly,
        )
        starting_index += stopping_index

    return integral_poly


@lru_cache(maxsize=None)
def get_B_matrix(cutoff, real_dtype):
    real_dtype = np.dtype(real_dtype)
    B_matrix = []

    for row_idx in range(cutoff):
        row_list = []
        for col_idx in range(row_idx + 1):
            row_list.extend(B(row_idx, col_idx, real_dtype.str).tolist())

        B_matrix.extend(row_list)

    return np.asarray(B_matrix, dtype=real_dtype)


@nb.njit(cache=True)
def get_interval(mean_position, poly):
    lower = mean_position - 5.0
    upper = mean_position + 5.0

    almost_0 = integral(poly, lower)
    almost_1 = integral(poly, upper)

    lower_tol = 1e-10
    upper_tol = 1.0 - lower_tol

    while almost_0 > lower_tol:
        lower -= 5.0
        almost_0 = integral(poly, lower)

    while almost_1 < upper_tol:
        upper += 5.0
        almost_1 = integral(poly, upper)

    return lower, upper, almost_0, almost_1


@lru_cache(maxsize=None)
def B(n, m, real_dtype_str):
    real_dtype = np.dtype(real_dtype_str)

    if n > m:
        return B(m, n, real_dtype_str)

    normalizer = np.sqrt(factorial(n) * factorial(m) * 2 ** (n + m) * np.pi)

    sum_ = np.zeros(1, dtype=real_dtype)

    for k in range(n):
        h1 = np.asarray(
            hermite(n - k).c,
            dtype=real_dtype,
        )
        h2 = np.asarray(
            hermite(m - k - 1).c,
            dtype=real_dtype,
        )

        term = np.polymul(h1, h2)
        term *= real_dtype.type(2**k / factorial(n - k))
        sum_ = np.polyadd(sum_, term).astype(real_dtype, copy=False)

    sum_ *= real_dtype.type(factorial(n) / normalizer)

    if n == m:
        return sum_

    # IMPORTANT: make a copy before scaling. ``hermite`` is cached above, and
    # ``np.asarray(..., dtype=real_dtype)`` may return a view of the cached
    # orthopoly1d coefficient array. An in-place ``*=`` would therefore corrupt
    # the cached Hermite polynomial and all subsequent B(n, m) values.
    extra = np.array(
        hermite(m - n - 1).c,
        dtype=real_dtype,
        copy=True,
    )
    extra *= real_dtype.type(2**n * factorial(n) / normalizer)

    return np.polyadd(extra, sum_).astype(real_dtype, copy=False)
