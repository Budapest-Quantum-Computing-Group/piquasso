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

"""
Efficient marginal photon-number sampling for a subset of modes.

When a :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement` only
requests ``k`` of the ``d`` output modes, sampling the full ``d``-mode boson
sampler and discarding the rest is wasteful for small ``k``. Instead the marginal
distribution over the requested modes can be computed directly.

The marginal probabilities are obtained with the Laguerre-polynomial generating
function (valid for collided Fock inputs). Let the occupied input modes be
``a = 1 .. s`` with occupations ``r_a`` (``sum_a r_a = n``), and let the requested
output modes be ``T = (l_1, .., l_k)``. With ``V[a, j] = conj(U[l_j, a])`` and
``f_a(z) = sum_j V[a, j] z_j``,

    P(z, w) = prod_a L_{r_a}( -f_a(z) * conj_coeff(f_a(w)) ),

where ``L_r`` is the degree-``r`` Laguerre polynomial. The marginal probability of
observing ``y`` on the requested modes is

    Pr(Y = y) = sum_{alpha >= y, |alpha| <= n}
                    alpha! P_{alpha, alpha} prod_j C(alpha_j, y_j) (-1)^{alpha_j - y_j},

with ``P_{alpha, alpha}`` the coefficient of ``z^alpha w^alpha`` in ``P``. The
construction runs in ``O(n^{2k + 1})`` for fixed ``k``, so it is preferable to the
full-sampler-then-discard approach only while ``k`` is small (see
:func:`prefer_marginal_sampling`).
"""

from math import comb, factorial, prod
from typing import List, Tuple

import numpy as np


def _configurations(modes: int, total: int):
    """All length-``modes`` non-negative integer tuples summing to ``total``."""
    if modes == 0:
        if total == 0:
            yield ()
        return
    if modes == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _configurations(modes - 1, total - first):
            yield (first,) + rest


def _add(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


def _unit(j: int, k: int) -> tuple:
    return tuple(1 if i == j else 0 for i in range(k))


def _multiply(left: dict, right: dict) -> dict:
    """Multiply two polynomials in the (z, w) multi-index dictionary form."""
    out: dict = {}
    for (lz, lw), lc in left.items():
        for (rz, rw), rc in right.items():
            key = (_add(lz, rz), _add(lw, rw))
            out[key] = out.get(key, 0j) + lc * rc
    return out


def _generating_polynomial(V: np.ndarray, occupations: List[int], k: int) -> dict:
    """Build ``P(z, w) = prod_a L_{r_a}(-f_a(z) conj_coeff(f_a(w)))`` as a dict
    mapping ``(alpha, beta)`` multi-index pairs to coefficients."""
    zero = ((0,) * k, (0,) * k)
    poly = {zero: 1.0 + 0j}

    for a, r_a in enumerate(occupations):
        # g_a = -sum_{j, j'} V[a, j] conj(V[a, j']) z_j w_{j'}
        g: dict = {}
        for j in range(k):
            for jp in range(k):
                key = (_unit(j, k), _unit(jp, k))
                g[key] = g.get(key, 0j) - V[a, j] * np.conj(V[a, jp])

        # L_{r_a}(g) = sum_{m=0}^{r_a} C(r_a, m) (-1)^m / m! * g^m
        factor: dict = {}
        g_power = {zero: 1.0 + 0j}
        for m in range(r_a + 1):
            coeff = comb(r_a, m) * ((-1) ** m) / factorial(m)
            for key, value in g_power.items():
                factor[key] = factor.get(key, 0j) + coeff * value
            if m < r_a:
                g_power = _multiply(g_power, g)

        poly = _multiply(poly, factor)

    return poly


def marginal_distribution(
    interferometer: np.ndarray,
    input_occupation: List[int],
    modes: Tuple[int, ...],
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """Return the marginal photon-number distribution over ``modes``.

    Args:
        interferometer: the (lossless, unitary) interferometer matrix.
        input_occupation: the Fock input occupation per mode.
        modes: the requested output modes.

    Returns:
        ``(outcomes, probabilities)``, parallel lists of the possible ``modes``
        photon-number tuples and their probabilities.
    """
    occupied = [i for i, r in enumerate(input_occupation) if r > 0]
    occupations = [int(input_occupation[i]) for i in occupied]
    n = sum(occupations)
    k = len(modes)

    V = np.array(
        [[np.conj(interferometer[modes[j], a]) for j in range(k)] for a in occupied],
        dtype=complex,
    )

    poly = _generating_polynomial(V, occupations, k)
    diagonal = {alpha: c for (alpha, beta), c in poly.items() if alpha == beta}

    outcomes: List[Tuple[int, ...]] = []
    probabilities: List[float] = []
    for total in range(n + 1):
        for y in _configurations(k, total):
            probability = 0j
            for alpha, coeff in diagonal.items():
                if any(alpha[j] < y[j] for j in range(k)):
                    continue
                term = prod(factorial(alpha[j]) for j in range(k)) * coeff
                term *= prod(
                    comb(alpha[j], y[j]) * ((-1) ** (alpha[j] - y[j])) for j in range(k)
                )
                probability += term
            outcomes.append(y)
            probabilities.append(probability.real)

    probs = np.array(probabilities, dtype=float)
    # guard against tiny negative round-off, then renormalise
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return outcomes, probs


def sample_marginal(
    interferometer: np.ndarray,
    input_occupation: List[int],
    modes: Tuple[int, ...],
    shots: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """Draw ``shots`` samples from the marginal distribution over ``modes``."""
    outcomes, probs = marginal_distribution(interferometer, input_occupation, modes)
    indices = rng.choice(len(outcomes), size=shots, p=probs)
    return [outcomes[i] for i in indices]


def prefer_marginal_sampling(n: int, k: int, d: int, shots: int) -> bool:
    """Heuristic FLOP comparison: prefer the marginal algorithm over
    full-sampling-then-discard when measuring a strict subset of modes and the
    ``O(n^{2k+1})`` marginal build is cheaper than drawing ``shots`` full samples.

    The crossover constant is chosen from the benchmark in
    ``benchmark/sampling_marginal_benchmark.py``.
    """
    if not (0 < k < d):
        return False
    # cost of building the marginal once, vs. the per-shot full-sampler cost
    marginal_cost = (n + 1) ** (2 * k + 1)
    discard_cost = shots * d * (2**d)
    return marginal_cost <= discard_cost
