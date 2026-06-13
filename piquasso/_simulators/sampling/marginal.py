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

r"""Exact joint marginal photon-number sampling for Boson Sampling.

Given a Fock input state :math:`\ket{\mathbf{r}}` with :math:`n = \sum_b r_b`
photons interfering on a unitary :math:`U`, this module samples the joint
photon-number distribution restricted to a subset
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

The diagonal coefficients are obtained by evaluating :math:`P` on the tensor
grid of :math:`(n+1)`-th roots of unity (scaled to a numerically stable radius,
see below) and recovering them with a single :math:`2k`-dimensional FFT. Only
the diagonal slice is kept, and only entries with total degree
:math:`|\alpha| \leq n` are nonzero, so just :math:`\binom{n + k}{k}` scaled
coefficients :math:`\alpha! P_{\alpha, \alpha}` carry information, far fewer than
the full :math:`(n+1)^k` table.

Sampling is then done mode by mode via the chain rule

.. math::
    \Pr(y_1, \dots, y_k)
    = \Pr(y_1) \, \Pr(y_2 \mid y_1) \cdots \Pr(y_k \mid y_1, \dots, y_{k-1}).

The marginal of the first :math:`\ell` measured modes is exactly the
coefficient table restricted to its first :math:`\ell` axes with the trailing
indices set to zero, because setting :math:`z_j = w_j = 0` for :math:`j > \ell`
in :math:`P` recovers the moment polynomial of the first :math:`\ell` modes.
Conditioning on an outcome :math:`y_j` is a single contraction of the
corresponding axis with row :math:`y_j` of the signed binomial matrix
:math:`W_{y, \alpha} = \binom{\alpha}{y} (-1)^{\alpha - y}`. Drawing one mode at
a time this way keeps everything in terms of the stored coefficients, supports
postselection on measured modes by fixing the corresponding outcome, and never
materializes the full joint distribution.

The contour radius matters. On the unit torus the coefficient extraction is
catastrophically ill-conditioned for :math:`n \gtrsim 14`: the polynomial
reaches magnitude of order :math:`(1 + k)^n` on the grid while the diagonal
coefficients are tiny, and the subsequent multiplication by :math:`\alpha!`
amplifies the FFT roundoff into errors that can exceed the probabilities
themselves. Evaluating instead on a torus of radius :math:`\rho = \sqrt{n / k}`
and dividing the extracted coefficient :math:`P_{\alpha, \alpha}` by
:math:`\rho^{2 |\alpha|}` balances the growth against the rescaling (this places
the contour near the saddle point) and restores machine precision to large
:math:`n`.

References:
    - S. Aaronson and A. Arkhipov, *The Computational Complexity of Linear
      Optics*, Theorem 12.6, https://arxiv.org/abs/1011.3245 (collision-free
      marginals via Gurvits's algorithm; the present method generalizes to
      collided inputs).
    - W. Roga and M. Takeoka, *Classical simulation of boson sampling with
      sparse output*, https://arxiv.org/abs/1904.05494 (use of k-mode
      marginals).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.special import comb, eval_laguerre, factorial


def _generate_marginal_samples(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
    shots: int,
    rng: np.random.Generator,
    postselect_modes: Tuple[int, ...] = (),
    postselect_photons: Tuple[int, ...] = (),
) -> Optional[List[Tuple[int, ...]]]:
    """Samples the measured modes from the exact joint marginal distribution.

    The measured modes are sampled one at a time via the chain rule, reusing a
    compact table of the scaled diagonal moment coefficients. Shots that share a
    prefix are conditioned together, so the work scales with the number of
    distinct outcomes rather than with the number of shots. Postselection on a
    subset of the measured modes is supported by fixing those outcomes.

    Returns ``None`` when the conditionals fail the numerical sanity check, in
    which case the caller should fall back to full sampling.
    """
    initial_state = np.asarray(initial_state, dtype=int)
    modes_array = np.asarray(modes, dtype=int)

    k = len(modes_array)
    n = int(np.sum(initial_state))

    coefficients = _calculate_scaled_diagonal_coefficients(
        interferometer, initial_state, modes_array, n
    )

    postselected_axes = _resolve_postselected_axes(
        modes, postselect_modes, postselect_photons
    )

    binomial_matrix = _signed_binomial_matrix(n + 1)

    return _sample_chain_rule(
        coefficients, binomial_matrix, k, shots, rng, postselected_axes
    )


def _sample_chain_rule(
    coefficients: np.ndarray,
    binomial_matrix: np.ndarray,
    k: int,
    shots: int,
    rng: np.random.Generator,
    postselected_axes: Dict[int, int],
) -> Optional[List[Tuple[int, ...]]]:
    """Draws ``shots`` samples by walking the measured modes one at a time.

    Shots are grouped by their already-sampled prefix so the per-axis
    conditioning is performed once per distinct prefix.
    """
    # Each group is (number_of_shots_in_group, conditioned_coefficients, prefix).
    groups: List[Tuple[int, np.ndarray, Tuple[int, ...]]] = [(shots, coefficients, ())]

    for axis in range(k):
        next_groups: List[Tuple[int, np.ndarray, Tuple[int, ...]]] = []

        for count, conditioned, prefix in groups:
            # Marginalize the not-yet-sampled modes (trailing-zero property).
            trailing = conditioned.ndim - 1
            reduced = conditioned[(slice(None),) + (0,) * trailing]

            probabilities = np.clip((binomial_matrix @ reduced).real, 0.0, None)
            total = probabilities.sum()

            if total <= 0.0:
                return None

            probabilities = probabilities / total

            if axis in postselected_axes:
                photons = postselected_axes[axis]
                if photons >= len(probabilities) or probabilities[photons] <= 0.0:
                    # This branch cannot satisfy the postselection; drop its
                    # shots (rejection) rather than aborting the whole sample.
                    continue
                outcome_counts = np.zeros(len(probabilities), dtype=int)
                outcome_counts[photons] = count
            else:
                outcome_counts = rng.multinomial(count, probabilities)

            for value, value_count in enumerate(outcome_counts):
                if value_count == 0:
                    continue
                next_conditioned = np.tensordot(
                    binomial_matrix[value], conditioned, axes=([0], [0])
                )
                next_groups.append(
                    (int(value_count), next_conditioned, prefix + (value,))
                )

        groups = next_groups

    samples: List[Tuple[int, ...]] = []
    for count, _, prefix in groups:
        samples.extend([prefix] * count)

    rng.shuffle(samples)

    return samples


def _resolve_postselected_axes(
    modes: Tuple[int, ...],
    postselect_modes: Tuple[int, ...],
    postselect_photons: Tuple[int, ...],
) -> Dict[int, int]:
    """Maps measured-mode positions to their required photon numbers."""
    mode_to_axis = {mode: axis for axis, mode in enumerate(modes)}
    return {
        mode_to_axis[mode]: photons
        for mode, photons in zip(postselect_modes, postselect_photons)
        if mode in mode_to_axis
    }


def _calculate_scaled_diagonal_coefficients(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: np.ndarray,
    n: int,
) -> np.ndarray:
    r"""The scaled diagonal moment coefficients :math:`\alpha! P_{\alpha,\alpha}`.

    Returns an array of shape :math:`(n + 1)^{\times k}`; only the entries with
    total degree :math:`|\alpha| \leq n` are nonzero. These are exactly the
    quantities the chain-rule sampler consumes.
    """
    k = len(modes)

    if n == 0:
        return np.ones(shape=(1,) * k, dtype=np.float64)

    occupied = np.where(initial_state > 0)[0]
    occupations = initial_state[occupied]

    submatrix = np.asarray(interferometer, dtype=np.complex128)[np.ix_(modes, occupied)]

    N = n + 1

    # Balanced contour radius; see the module docstring.
    rho = np.sqrt(n / k)

    diagonal = _extract_diagonal_coefficients(submatrix, occupations, k, N, rho)

    total_degrees = np.sum(np.indices((N,) * k), axis=0)

    diagonal = np.where(
        total_degrees <= n,
        diagonal.real / rho ** (2.0 * total_degrees),
        0.0,
    )

    coefficients = diagonal
    for axis in range(k):
        shape = [1] * k
        shape[axis] = N
        coefficients = coefficients * factorial(np.arange(N)).reshape(shape)

    return coefficients


def _extract_diagonal_coefficients(
    submatrix: np.ndarray, occupations: np.ndarray, k: int, N: int, rho: float
) -> np.ndarray:
    r"""Diagonal coefficients :math:`P_{\alpha,\alpha}` of the moment polynomial.

    The polynomial is evaluated on the tensor grid of :math:`N`-th roots of unity
    scaled by ``rho``, its coefficients are recovered with a single
    :math:`2k`-dimensional FFT, and the diagonal :math:`\alpha = \beta` slice is
    returned (still scaled by :math:`\rho^{2 |\alpha|}`, which the caller undoes).
    """
    roots_of_unity = rho * np.exp(2j * np.pi * np.arange(N) / N)

    grid = np.stack(
        [m.ravel() for m in np.meshgrid(*([roots_of_unity] * k), indexing="ij")],
        axis=-1,
    )

    h = grid @ submatrix
    g = grid @ np.conj(submatrix)

    values = np.ones(shape=(len(grid), len(grid)), dtype=np.complex128)
    for b in range(len(occupations)):
        argument = -np.multiply.outer(h[:, b], g[:, b])
        values *= eval_laguerre(occupations[b], argument)

    values = values.reshape((N,) * (2 * k))

    coefficients = np.fft.fftn(values) / N ** (2 * k)

    diagonal_indices = tuple(np.indices((N,) * k))

    return coefficients[diagonal_indices + diagonal_indices]


def _signed_binomial_matrix(N: int) -> np.ndarray:
    r"""The matrix :math:`W_{y, \alpha} = \binom{\alpha}{y} (-1)^{\alpha - y}`."""
    indices = np.arange(N)
    return comb(indices[None, :], indices[:, None]) * np.where(
        indices[None, :] >= indices[:, None],
        (-1.0) ** (indices[None, :] - indices[:, None]),
        0.0,
    )


def _marginal_strategy_is_preferred(
    number_of_particles: int,
    number_of_measured_modes: int,
    number_of_occupied_modes: int,
    number_of_modes: int,
    shots: int,
) -> bool:
    """Decides whether sampling from the exact marginal distribution is cheaper
    than generating full samples and discarding the unmeasured modes.

    The decision compares estimated worst-case floating point operation counts
    of the two strategies and errs on the side of the existing full sampler. The
    transient FFT buffer is also capped to bound memory use.
    """
    n = number_of_particles
    k = number_of_measured_modes
    s = max(number_of_occupied_modes, 1)

    grid_points = float(n + 1) ** (2 * k)

    if grid_points > _MAX_FFT_POINTS:
        return False

    # Coefficient table: one 2k-dimensional FFT over (n+1)^{2k} points plus the
    # per-input-mode Laguerre evaluations. Sampling: O(k (n+1)^2) per distinct
    # outcome, bounded by the number of shots.
    marginal_flops = grid_points * (np.log2(grid_points) + s) + shots * k * (n + 1) ** 2

    # Each full Clifford & Clifford sample evaluates permanents whose dominant
    # cost scales like n * d * 2^n.
    sampler_flops = shots * n * number_of_modes * 2.0 ** min(n, 48)

    return marginal_flops < sampler_flops


# Beyond this transient FFT buffer size (~100 MB at 16 bytes per complex point)
# the marginal strategy is never chosen, regardless of the FLOP estimate.
_MAX_FFT_POINTS = 6_000_000
