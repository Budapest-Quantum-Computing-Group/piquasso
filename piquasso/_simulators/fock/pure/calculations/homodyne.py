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

from piquasso.instructions.measurements import HomodyneMeasurement
from piquasso.api.result import Result
from piquasso.api.exceptions import NotImplementedCalculation

from piquasso._math.fock import get_fock_space_basis

from ..state import PureFockState

from scipy.special import hermite, factorial
from scipy.optimize import brentq

import numpy as np
import numba as nb

from functools import lru_cache


hermite = lru_cache(maxsize=None)(hermite)


def homodyne_measurement(
    state: PureFockState, instruction: HomodyneMeasurement, shots: int
) -> Result:
    modes = instruction.modes

    phi = instruction.params["phi"]

    if state._config.validate and not np.isclose(phi, 0.0):
        raise NotImplementedCalculation(
            "'HomodyneMeasurement' with nonzero rotation angle is not yet supported."
        )

    reduced_state = state.reduced(modes=modes)

    cutoff = state._config.cutoff
    hbar = state._config.hbar

    sqrt_hbar = np.sqrt(hbar)

    reduced_state.normalize()

    density_matrix = reduced_state.density_matrix

    rng = state._config.rng
    small_d = len(modes)

    mean_positions = np.empty(shape=small_d)

    for idx in range(small_d):
        mean_positions[idx] = state.mean_position(modes[idx]) / sqrt_hbar

    uniforms = rng.uniform(size=(shots, small_d))

    samples = np.empty(shape=(shots, small_d))

    B_matrix = get_B_matrix(cutoff)

    if small_d == 1:
        _homodyne_measurement_one_mode(
            density_matrix,
            shots,
            mean_positions[0],
            B_matrix,
            samples,
            uniforms[:, 0],
        )

    else:
        density_matrix_mode_0 = reduced_state.reduced(modes=(0,)).density_matrix

        spaces = []

        for current_d in range(2, state.d + 1):
            spaces.append(get_fock_space_basis(d=current_d, cutoff=cutoff))

        hermites_size = cutoff * (cutoff + 1) // 2

        hermites = np.empty(shape=hermites_size)

        starting_index = 0

        for idx in range(cutoff):
            hermite_polynomial_size = idx + 1
            hermite_polynomial_coeffs = np.array(hermite(idx))
            hermites[starting_index : starting_index + hermite_polynomial_size] = (
                hermite_polynomial_coeffs
            )
            starting_index += hermite_polynomial_size

        _homodyne_measurement_one_mode(
            density_matrix_mode_0,
            shots=shots,
            mean_position=mean_positions[0],
            B_matrix=B_matrix,
            samples=samples[:, 0],
            uniforms=uniforms[:, 0],
        )

        reduced_density_matrices = []
        for current_d in range(2, reduced_state.d + 1):
            reduced_state_to_current = reduced_state.reduced(
                modes=tuple(range(current_d))
            )
            reduced_density_matrices.append(reduced_state_to_current.density_matrix)

        new_density_matrix = np.empty_like(density_matrix_mode_0)

        _do_sample_homodyne_multimodes(
            shots,
            samples,
            new_density_matrix,
            reduced_density_matrices,
            spaces,
            hermites,
            mean_positions,
            B_matrix,
            uniforms,
        )

    return Result(state=state, samples=sqrt_hbar * samples)


def _do_sample_homodyne_multimodes(
    shots,
    samples,
    new_density_matrix,
    reduced_density_matrices,
    spaces,
    hermites,
    mean_positions,
    B_matrix,
    uniforms,
):
    for i in range(shots):
        _homodyne_measurement_multimode_sample(
            samples[i],
            new_density_matrix,
            reduced_density_matrices,
            spaces,
            hermites,
            mean_positions,
            B_matrix,
            uniforms[i],
        )


def _homodyne_measurement_multimode_sample(
    positions,
    new_density_matrix,
    reduced_density_matrices,
    spaces,
    hermites,
    mean_positions,
    B_matrix,
    uniforms,
):
    d = len(reduced_density_matrices) + 1
    cutoff = new_density_matrix.shape[0]

    for current_d in range(2, d + 1):
        density_matrix = reduced_density_matrices[current_d - 2]

        new_density_matrix.fill(0.0)

        space = spaces[current_d - 2]

        hermite_terms = get_hermite_terms(hermites, positions, space, current_d, cutoff)

        new_density_matrix = _get_density_matrix_on_next_mode(
            new_density_matrix, density_matrix, space, hermite_terms
        )

        _homodyne_measurement_one_mode(
            new_density_matrix,
            shots=1,
            mean_position=mean_positions[current_d - 1],
            B_matrix=B_matrix,
            samples=positions[current_d - 1 :],
            uniforms=uniforms[current_d - 1 :],
        )


def _homodyne_measurement_one_mode(
    density_matrix, shots, mean_position, B_matrix, samples, uniforms
):
    poly = get_integral_poly(density_matrix, B_matrix)

    lower, upper, almost_0, almost_1 = get_interval(mean_position, poly)

    for idx in range(shots):
        inverse_sample = uniforms[idx] * (almost_1 - almost_0) + almost_0

        sample = brentq(
            integral_m_inverse_sample, a=lower, b=upper, args=(poly, inverse_sample)
        )

        samples[idx] = sample


@nb.njit(cache=True)
def _get_density_matrix_on_next_mode(
    new_density_matrix, density_matrix, space, hermite_terms
):
    for row_idx in range(density_matrix.shape[0]):
        row_occupation_number = space[row_idx]

        row_term = hermite_terms[row_idx]

        small_row_idx = row_occupation_number[-1]

        for col_idx in range(row_idx):
            col_occupation_number = space[col_idx]

            col_term = hermite_terms[col_idx]

            small_col_idx = col_occupation_number[-1]

            row_col_term = row_term * col_term

            new_density_matrix[small_row_idx, small_col_idx] += (
                density_matrix[row_idx, col_idx] * row_col_term
            )

            new_density_matrix[small_col_idx, small_row_idx] += (
                density_matrix[col_idx, row_idx] * row_col_term
            )

        diagonal = density_matrix[row_idx, row_idx] * row_term**2

        new_density_matrix[small_row_idx, small_row_idx] += diagonal

    new_density_matrix /= np.trace(new_density_matrix)

    return new_density_matrix


@nb.njit(cache=True)
def polyeval(p, x):
    y = 0.0

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
def get_integral_poly(density_matrix, B_matrix):
    size = density_matrix.shape[0]

    integral_poly = np.array([0.0])

    starting_index = 0

    for row_idx in range(size):
        for col_idx in range(row_idx):
            stopping_index = row_idx + col_idx
            sliced_I = B_matrix[starting_index : starting_index + stopping_index]

            poly_to_add = sliced_I * 2 * np.real(density_matrix[row_idx, col_idx])
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
def get_B_matrix(cutoff):
    B_matrix = []

    for row_idx in range(cutoff):
        row_list = []
        for col_idx in range(row_idx + 1):
            row_list.extend(np.array(B(row_idx, col_idx)).tolist())

        B_matrix.extend(row_list)

    return np.array(B_matrix)


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


@nb.njit(cache=True)
def _get_hermite_vals(hermites, current_d, positions, cutoff):
    hermite_vals = np.empty(shape=(cutoff, current_d - 1))

    starting_index = 0
    for idx in range(cutoff):
        size = idx + 1
        hermite_polynomial_coeffs = hermites[starting_index : starting_index + size]
        starting_index += size
        for jdx in range(current_d - 1):
            hermite_vals[idx, jdx] = polyeval(hermite_polynomial_coeffs, positions[jdx])

    return hermite_vals


@nb.njit(cache=True)
def _do_get_hermite_terms(space, hermite_vals, current_d):
    hermite_terms = np.empty(shape=space.shape[0])

    for idx in range(space.shape[0]):
        occupation_number = space[idx]

        term = 1.0
        for jdx in range(current_d - 1):
            term *= hermite_vals[occupation_number[jdx], jdx]

        hermite_terms[idx] = term

    return hermite_terms


@nb.njit(cache=True)
def get_hermite_terms(hermites, positions, space, current_d, cutoff):
    hermite_vals = _get_hermite_vals(hermites, current_d, positions, cutoff)

    return _do_get_hermite_terms(space, hermite_vals, current_d)


@lru_cache(maxsize=None)
def B(n, m):
    if n > m:
        return B(m, n)

    normalizer = np.sqrt(factorial(n) * factorial(m) * 2 ** (n + m) * np.pi)

    sum_ = np.poly1d([0.0])

    for k in range(n):
        sum_ += hermite(n - k) * hermite(m - k - 1) * 2**k / factorial(n - k)

    sum_ = sum_ * factorial(n) / normalizer

    if n == m:
        return sum_

    return hermite(m - n - 1) * 2**n * factorial(n) / normalizer + sum_
