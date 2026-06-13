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

r"""Exact joint marginal photon-number distribution for Boson Sampling.

Given a Fock input state :math:`\ket{\mathbf{r}}` with :math:`n = \sum_b r_b`
photons interfering on a unitary :math:`U`, this module computes the *exact*
joint photon-number distribution restricted to a subset
:math:`T = (\ell_1, \dots, \ell_k)` of output modes, in time polynomial in
:math:`n` for fixed :math:`k`, i.e., without enumerating the exponentially
many outcomes on the unmeasured modes.

The algorithm is based on the (multivariate) probability generating function

.. math::
    G(\mathbf{x})
    = \bra{\mathbf{r}} \Gamma(U)^\dagger
      \prod_{j=1}^{k} x_j^{\hat{n}_{\ell_j}} \Gamma(U) \ket{\mathbf{r}}
    = \sum_{\mathbf{y}} \Pr(\mathbf{y}) \, \mathbf{x}^{\mathbf{y}}.

Using the normally ordered expansion
:math:`x^{\hat{n}} = \sum_{p} \frac{(x-1)^p}{p!} a^{\dagger p} a^p` and the
identity :math:`\bra{r} e^{\beta a^\dagger} e^{\gamma a} \ket{r} =
L_r(-\beta\gamma)` (:math:`L_r` being the Laguerre polynomial of degree
:math:`r`), all normally ordered moments are encoded in the polynomial

.. math::
    P(\mathbf{z}, \mathbf{w})
    = \prod_{b} L_{r_b}\left( -h_b(\mathbf{z}) \, g_b(\mathbf{w}) \right),
    \qquad
    h_b(\mathbf{z}) = \sum_{j} U_{\ell_j b} z_j,
    \quad
    g_b(\mathbf{w}) = \sum_{j} \overline{U_{\ell_j b}} w_j,

whose *diagonal* coefficients :math:`P_{\alpha, \alpha}` (of
:math:`\mathbf{z}^\alpha \mathbf{w}^\alpha`) yield the probabilities via

.. math::
    \Pr(\mathbf{y})
    = \sum_{\alpha \geq \mathbf{y}} \alpha! \, P_{\alpha, \alpha}
      \prod_{j=1}^k \binom{\alpha_j}{y_j} (-1)^{\alpha_j - y_j}.

Instead of expanding :math:`P` symbolically (which is prohibitively slow), the
implementation evaluates :math:`P` *numerically* on the tensor grid of
:math:`(n+1)`-th roots of unity, recovers all coefficients with a single
:math:`2k`-dimensional FFT, reads off the diagonal slice, and contracts with
signed binomial matrices. This yields the *entire* joint distribution on the
measured modes at once using

.. math::
    O\big( (n+1)^{2k} (s + k) \big)

floating point operations (:math:`s` being the number of occupied input
modes), fully vectorized, after which any number of shots can be drawn at
negligible cost. The unmeasured modes never enter the computation.

For comparison, the textbook alternative of sampling *all* modes and
discarding the unmeasured ones costs
:math:`O(\text{shots} \cdot n 2^n)`-ish with Clifford-Clifford-type samplers,
which is preferable for large :math:`k` or very few shots; the simulator
chooses between the two strategies based on a calibrated cost model (see
:func:`marginal_strategy_is_preferred`).

References:
    - S. Aaronson and A. Arkhipov, *The Computational Complexity of Linear
      Optics*, Theorem 12.6, https://arxiv.org/abs/1011.3245 (collision-free
      marginals via Gurvits's algorithm; the present method generalizes to
      collided inputs).
    - W. Roga and M. Takeoka, *Classical simulation of boson sampling with
      sparse output*, https://arxiv.org/abs/1904.05494 (use of k-mode
      marginals).
"""

from typing import List, Optional, Tuple

import numpy as np

from scipy.special import comb, factorial

# Hard cap on the number of grid points (n+1)^{2k} so that the FFT buffer
# (16 bytes/point) stays below ~100 MB and runtime below ~1s.
_MAX_GRID_POINTS = 6_000_000

# Calibrated cost-model constants (seconds); see `marginal_strategy_is_preferred`.
_MARGINAL_SECONDS_PER_POINT = 3e-8
_SAMPLER_SECONDS_PER_SHOT_OVERHEAD = 2e-4
_SAMPLER_SECONDS_PER_FLOP = 6e-9


def calculate_marginal_particle_number_distribution(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
) -> np.ndarray:
    r"""The exact joint photon-number distribution on a subset of output modes.

    Args:
        interferometer: The (unitary) interferometer matrix.
        initial_state: The input Fock basis state (occupation numbers).
        modes: The measured output modes.

    Returns:
        Array of shape :math:`(n+1)^{\times k}`, where the entry at index
        :math:`\mathbf{y} = (y_1, \dots, y_k)` is the probability of detecting
        :math:`y_j` photons in mode ``modes[j]``.
    """
    initial_state = np.asarray(initial_state, dtype=int)
    modes_array = np.asarray(modes, dtype=int)

    k = len(modes_array)
    n = int(np.sum(initial_state))

    if n == 0:
        probabilities = np.zeros(shape=(1,) * k, dtype=np.float64)
        probabilities[(0,) * k] = 1.0
        return probabilities

    occupied = np.where(initial_state > 0)[0]
    occupations = initial_state[occupied]

    submatrix = np.asarray(interferometer, dtype=np.complex128)[
        np.ix_(modes_array, occupied)
    ]

    N = n + 1

    # The coefficients are extracted on a torus of radius `rho` balancing the
    # magnitude of the polynomial against the rescaling of the coefficients,
    # which keeps the FFT-based extraction numerically stable for large `n`.
    rho = np.sqrt(max(n, 1) / k)

    diagonal_coefficients = _calculate_diagonal_coefficients(
        submatrix, occupations, k, N, rho
    )

    # Discard |alpha| > n entries (exactly zero in theory, noise numerically),
    # and undo the radius rescaling of the coefficients.
    alphas = np.indices((N,) * k)
    total_degrees = np.sum(alphas, axis=0)
    diagonal_coefficients = np.where(
        total_degrees <= n,
        diagonal_coefficients.real / rho ** (2.0 * total_degrees),
        0.0,
    )

    # c_alpha = alpha! * P_{alpha, alpha}
    coefficients = diagonal_coefficients
    for axis in range(k):
        shape = [1] * k
        shape[axis] = N
        coefficients = coefficients * factorial(np.arange(N)).reshape(shape)

    # Pr(y) = sum_{alpha >= y} c_alpha prod_j C(alpha_j, y_j) (-1)^(alpha_j-y_j)
    indices = np.arange(N)
    binomial_matrix = comb(indices[None, :], indices[:, None]) * np.where(
        indices[None, :] >= indices[:, None],
        (-1.0) ** (indices[None, :] - indices[:, None]),
        0.0,
    )

    probabilities = coefficients
    for axis in range(k):
        probabilities = np.moveaxis(
            np.tensordot(binomial_matrix, probabilities, axes=([1], [axis])), 0, axis
        )

    return probabilities


def _calculate_diagonal_coefficients(
    submatrix: np.ndarray, occupations: np.ndarray, k: int, N: int, rho: float
) -> np.ndarray:
    r"""Diagonal coefficients :math:`P_{\alpha,\alpha}` of the moment polynomial.

    The polynomial :math:`P(\mathbf{z}, \mathbf{w})` is evaluated on the full
    tensor grid of :math:`N`-th roots of unity scaled by ``rho``
    (:math:`N = n + 1`), its coefficients are recovered with a single
    :math:`2k`-dimensional FFT, and the diagonal :math:`\alpha = \beta` slice
    is returned. Note, that the returned coefficients are scaled by
    :math:`\rho^{2 |\alpha|}`, which the caller needs to undo.
    """
    roots_of_unity = rho * np.exp(2j * np.pi * np.arange(N) / N)

    # All grid points z in {roots}^k, flattened: shape (N^k, k).
    grid = np.stack(
        [m.ravel() for m in np.meshgrid(*([roots_of_unity] * k), indexing="ij")],
        axis=-1,
    )

    h = grid @ submatrix  # h_b(z) for every grid point z; shape (N^k, s)
    g = grid @ np.conj(submatrix)  # g_b(w) for every grid point w

    values = np.ones(shape=(len(grid), len(grid)), dtype=np.complex128)
    for b in range(len(occupations)):
        argument = -np.multiply.outer(h[:, b], g[:, b])
        values *= _eval_laguerre(occupations[b], argument)

    values = values.reshape((N,) * (2 * k))

    coefficients = np.fft.fftn(values) / N ** (2 * k)

    diagonal_indices = tuple(np.indices((N,) * k))

    return coefficients[diagonal_indices + diagonal_indices]


def _eval_laguerre(degree: int, x: np.ndarray) -> np.ndarray:
    """Laguerre polynomial of the given degree, elementwise on a complex array."""
    previous = np.ones_like(x)

    if degree == 0:
        return previous

    current = 1.0 - x

    for m in range(1, degree):
        previous, current = (
            current,
            ((2 * m + 1 - x) * current - m * previous) / (m + 1),
        )

    return current


def marginal_strategy_is_preferred(
    number_of_particles: int,
    number_of_measured_modes: int,
    number_of_occupied_modes: int,
    number_of_modes: int,
    shots: int,
) -> bool:
    """Decides whether computing the exact marginal distribution is preferable
    to generating full samples and discarding the unmeasured modes.

    The estimates below were calibrated against the built-in Clifford-Clifford
    type sampler. The decision errs on the side of the full sampler: the
    marginal strategy is chosen only when it is predicted to be clearly
    cheaper, and its memory footprint is hard-capped.
    """
    n = number_of_particles
    k = number_of_measured_modes

    grid_points = float(n + 1) ** (2 * k)

    if grid_points > _MAX_GRID_POINTS:
        return False

    marginal_seconds = (
        _MARGINAL_SECONDS_PER_POINT * grid_points * max(number_of_occupied_modes, 1)
    )

    sampler_seconds = shots * (
        _SAMPLER_SECONDS_PER_SHOT_OVERHEAD
        + _SAMPLER_SECONDS_PER_FLOP * n * number_of_modes * 2.0 ** min(n, 40)
    )

    return marginal_seconds < sampler_seconds


def generate_marginal_samples(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
    shots: int,
    rng: np.random.Generator,
) -> Optional[List[Tuple[int, ...]]]:
    """Generates samples on the specified output modes from the exact joint
    marginal distribution.

    Returns ``None`` when the computed distribution fails the built-in
    numerical sanity check, in which case the caller should fall back to full
    sampling.
    """
    probabilities = calculate_marginal_particle_number_distribution(
        interferometer, initial_state, modes
    )

    flat_probabilities = probabilities.ravel()

    # Numerical sanity check; on failure the caller falls back to the sampler.
    if (
        np.min(flat_probabilities) < -1e-7
        or abs(np.sum(flat_probabilities) - 1.0) > 1e-7
    ):
        return None

    flat_probabilities = np.clip(flat_probabilities, 0.0, None)
    flat_probabilities /= np.sum(flat_probabilities)

    choices = rng.choice(len(flat_probabilities), size=shots, p=flat_probabilities)

    outcomes = np.stack(np.unravel_index(choices, probabilities.shape), axis=-1)

    return [tuple(map(int, outcome)) for outcome in outcomes]
