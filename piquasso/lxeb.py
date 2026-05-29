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
Utilities for computing LXEB reference values in photonic quantum advantage schemes.

This module provides numerically stable routines for evaluating Haar-averaged
linear cross-entropy benchmarking (LXEB) reference values for photonic quantum
advantage experiments.

For more details, see
`https://arxiv.org/abs/2604.15258 <https://arxiv.org/abs/2604.15258>`_.
"""

import math


def _log_binom(n: int, k: int) -> float:
    """log(binomial(n, k))"""
    if k < 0 or k > n:
        return float("-inf")

    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_poch(a: float, k: int) -> float:
    """log((a)_k), where (a)_k is the rising Pochhammer symbol."""

    if k < 0:
        raise ValueError("k must be non-negative")

    if k == 0:
        return 0.0

    if a == 0.0:
        return float("-inf")

    return math.lgamma(a + k) - math.lgamma(a)


def _logsumexp(values):
    if not values:
        return float("-inf")

    vmax = max(values)

    if vmax == float("-inf"):
        return vmax

    return vmax + math.log(sum(math.exp(v - vmax) for v in values))


def lxe_ref_boson_sampling(n: int, m: int) -> float:
    """Computes the Haar-averaged LXEB reference value for Boson Sampling.

    Args:
        n (int): Number of photons. Must be non-negative.
        m (int): Number of optical modes. Must be positive.

    Returns:
        float: The Haar-averaged LXEB reference value.

    Raises:
        TypeError: If ``n`` or ``m`` is not an integer.
        ValueError: If ``n`` is negative or ``m`` is not positive.
    """

    if not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("n and m must be integers")

    if n < 0:
        raise ValueError("n must be non-negative")

    if m <= 0:
        raise ValueError("m must be positive")

    log_terms = []

    for r in range(n // 2 + 1):
        log_term = (
            math.log(2 * n - 4 * r + 1)
            - math.log(2 * n - 2 * r + 1)
            - r * math.log(16.0)
            + _log_binom(2 * r, r)
            - _log_binom(2 * n - 2 * r, n - r)
            - _log_poch(m / 2.0, r)
            - _log_poch((m + 1) / 2.0, n - r)
        )
        log_terms.append(log_term)

    overall_log = n * math.log(2.0) + math.lgamma(n + 1) + _logsumexp(log_terms)

    return math.exp(overall_log)


def lxe_ref_gaussian_boson_sampling(n: int, m: int, d: int) -> float:
    """Computes the Haar-averaged LXEB reference value for Gaussian boson sampling.

    Args:
        n (int): Total detected photon number. Must be non-negative.
        m (int): Number of optical modes. Must be positive.
        d (int): Input rank parameter. Must satisfy ``1 <= d <= m``.

    Returns:
        float: The Haar-averaged LXEB reference value. Returns ``0.0`` when
        ``n`` is odd.

    Raises:
        TypeError: If ``n``, ``m``, or ``d`` is not an integer.
        ValueError: If ``n`` is negative, ``m`` is not positive, or ``d`` does
        not satisfy ``1 <= d <= m``.
    """

    if not isinstance(n, int) or not isinstance(m, int) or not isinstance(d, int):
        raise TypeError("n, m, and d must be integers")

    if n < 0:
        raise ValueError("n must be non-negative")

    if m <= 0:
        raise ValueError("m must be positive")

    if not (1 <= d <= m):
        raise ValueError("d must satisfy 1 <= d <= m")

    if n % 2 == 1:
        return 0.0

    N = n // 2
    log_terms = []

    for j in range(N + 1):
        log_poch_dminus1 = _log_poch((d - 1) / 2.0, j)
        if log_poch_dminus1 == float("-inf"):
            continue

        log_term = (
            math.log(4 * N - 4 * j + 1)
            - math.log(4 * N - 2 * j + 1)
            + _log_binom(2 * j, j)
            + 2.0 * _log_binom(2 * N - 2 * j, N - j)
            + log_poch_dminus1
            + _log_poch(d / 2.0 + N, N - j)
            - _log_binom(4 * N - 2 * j, 2 * N - j)
            - _log_poch(m / 2.0, j)
            - _log_poch((m + 1) / 2.0, 2 * N - j)
        )
        log_terms.append(log_term)

    overall_log = (
        2.0 * math.lgamma(N + 1) - _log_poch(d / 2.0, N) + _logsumexp(log_terms)
    )

    return math.exp(overall_log)
