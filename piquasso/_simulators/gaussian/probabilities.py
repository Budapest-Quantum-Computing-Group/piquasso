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

import abc

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import numba as nb

from scipy.special import factorial

from piquasso._math.linalg import block_reduce_xpxp
from piquasso._math.torontonian import torontonian, loop_torontonian
from piquasso._math.fock import nb_get_fock_space_basis
from piquasso.api.connector import BaseConnector



@nb.njit(cache=True)
def _recurrence_fill_G(
    G: np.ndarray,
    basis_2d: np.ndarray,
    succ: np.ndarray,
    pred: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    max_weight: int,
    ell: int,
) -> None:
    r"""Fill the array G of renormalized Fock amplitudes using the recurrence relation
    from Eq. (45) of arXiv:2209.06069.

    The renormalized multidimensional Hermite polynomial satisfies:

    .. math::
        \tilde{G}_{k + \mathbf{1}_i} = \frac{1}{\sqrt{k_i + 1}} \left[
            b_i \tilde{G}_k + \sum_j \sqrt{k_j} A_{ij} \tilde{G}_{k - \mathbf{1}_j}
        \right],

    where :math:`\tilde{G}_k = G_0 \cdot G^A_k(b) / \sqrt{k!}`.

    Args:
        G: Flat 1D array of complex Fock amplitudes indexed by Fock space position.
           Must be pre-allocated with G[0] = 1.0.
        basis_2d: Array of shape (N, ell) of multi-indices, sorted by weight.
        succ: Precomputed successor table of shape (N, ell).
              succ[idx, i] = flat index of basis[idx] + e_i, or -1 if out of range.
        pred: Precomputed predecessor table of shape (N, ell).
              pred[idx, j] = flat index of basis[idx] - e_j, or -1 if k[j] == 0.
        A: The ell x ell complex symmetric Bargmann A matrix.
        b: The ell-dimensional complex Bargmann b vector.
        max_weight: Maximum total photon number to compute (= 2*(cutoff-1)).
        ell: Dimension of the multi-index (= 2*d for d modes).
    """
    for idx in range(len(basis_2d)):
        k = basis_2d[idx]

        # Weight check: only process elements that have successors in range.
        w = np.int64(0)
        for dim in range(ell):
            w += k[dim]
        if w >= max_weight:
            continue

        # G value at current index (flat index == idx since basis_2d is sorted by
        # weight and get_index_in_fock_space_array is the identity mapping here).
        Gk = G[idx]

        for i in range(ell):
            new_flat_idx = succ[idx, i]
            if new_flat_idx < 0:
                continue

            ki = k[i]
            val = b[i] * Gk
            for j in range(ell):
                kj = k[j]
                if kj > 0:
                    prev_flat_idx = pred[idx, j]
                    val += np.sqrt(np.float64(kj)) * A[i, j] * G[prev_flat_idx]

            G[new_flat_idx] = val / np.sqrt(np.float64(ki + 1))


@lru_cache(maxsize=64)
def _get_cached_recurrence_data(
    ell: int,
    max_weight: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build and cache the Fock-space basis, successor/predecessor tables, mixed-radix
    hashes, and (when the hash range is small enough) a dense hash→flat-index lookup
    array for a given ``(ell, max_weight)`` pair.

    This function is called once per unique ``(ell, max_weight)`` combination and its
    result is reused across all subsequent ``get_density_matrix`` calls, eliminating
    the dominant O(N·ell) Python-dict construction that previously dominated runtime.

    Index tables are built fully vectorised with NumPy:

    * A mixed-radix hash ``h(k) = k · powers`` (with ``powers[i] = radix**i``) maps
      every multi-index to a unique integer.
    * When ``max(h) < 50 000 000`` a dense array ``lookup[h] = flat_idx`` gives O(1)
      neighbour look-up per mode per element.
    * Otherwise a sorted-hash + ``np.searchsorted`` fallback is used (still much faster
      than the Python dict).

    Args:
        ell: Dimension of the 2d-mode multi-index space (``= 2 * d``).
        max_weight: Maximum total photon number across all ``ell`` modes
            (``= 2 * (cutoff - 1)``).

    Returns:
        basis_2d:   ``(N, ell)`` int64 array of multi-indices sorted by weight.
        succ:       ``(N, ell)`` int64 successor table (``-1`` if out of range).
        pred:       ``(N, ell)`` int64 predecessor table (``-1`` if ``k[j] == 0``).
        hashes:     ``(N,)`` int64 mixed-radix hash of each row in ``basis_2d``.
        powers:     ``(ell,)`` int64 radix powers used to compute hashes.
        dense_lookup: ``(max_hash+1,)`` int64 array mapping hash → flat index, or
            ``None`` when the hash range exceeds the memory threshold.
    """
    basis_2d: np.ndarray = nb_get_fock_space_basis(d=ell, cutoff=max_weight + 1).copy()
    n = len(basis_2d)
    radix = int(max_weight) + 2
    powers: np.ndarray = radix ** np.arange(ell, dtype=np.int64)
    hashes: np.ndarray = basis_2d @ powers          # shape (N,)
    weights: np.ndarray = basis_2d.sum(axis=1)      # shape (N,)

    succ: np.ndarray = np.full((n, ell), -1, dtype=np.int64)
    pred: np.ndarray = np.full((n, ell), -1, dtype=np.int64)

    max_hash = int(hashes.max())
    dense_lookup: Optional[np.ndarray] = None

    if max_hash < 50_000_000:
        # Dense O(1) lookup – fits comfortably in memory.
        dense_lookup = np.full(max_hash + 1, -1, dtype=np.int64)
        dense_lookup[hashes] = np.arange(n, dtype=np.int64)

        for i in range(ell):
            # Successors: add powers[i] to hash, valid only when weight < max_weight.
            new_h = hashes + powers[i]
            valid = (weights < max_weight) & (new_h <= max_hash)
            rows = np.where(valid)[0]
            succ[rows, i] = dense_lookup[new_h[rows]]

            # Predecessors: subtract powers[i], valid only when k[i] > 0.
            rows2 = np.where(basis_2d[:, i] > 0)[0]
            pred[rows2, i] = dense_lookup[hashes[rows2] - powers[i]]
    else:
        # Searchsorted fallback for large hash ranges (high d / high cutoff).
        sort_idx = np.argsort(hashes)
        sorted_hashes = hashes[sort_idx]

        for i in range(ell):
            new_h = hashes + powers[i]
            mask = weights < max_weight
            pos = np.searchsorted(sorted_hashes, new_h)
            pos_c = np.minimum(pos, n - 1)
            found = mask & (pos < n) & (sorted_hashes[pos_c] == new_h)
            succ[found, i] = sort_idx[pos[found]]

            rows2 = np.where(basis_2d[:, i] > 0)[0]
            new_h2 = hashes[rows2] - powers[i]
            pos2 = np.searchsorted(sorted_hashes, new_h2)
            pos2_c = np.minimum(pos2, n - 1)
            found2 = sorted_hashes[pos2_c] == new_h2
            pred[rows2[found2], i] = sort_idx[pos2[found2]]

    return basis_2d, succ, pred, hashes, powers, dense_lookup


def _compute_G_array_via_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    cutoff: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the full array of renormalized Fock amplitudes using the recurrence
    relation from Eq. (45) of arXiv:2209.06069.

    Successor/predecessor tables are retrieved from a per-process LRU cache keyed by
    ``(ell, max_weight)`` so they are built at most once per unique truncation, making
    repeated calls (the common case) pay only the O(N·ell) numba recurrence cost.

    Args:
        A: The 2d × 2d Bargmann A matrix.
        b: The 2d Bargmann b vector.
        cutoff: Fock space truncation cutoff.
        d: Number of modes.

    Returns:
        G: 1-D complex128 array of renormalized Fock amplitudes indexed by the
           position of the 2d-mode multi-index in the weight-sorted basis.
        basis_2d: The corresponding ``(N, 2d)`` multi-index array.
    """
    ell = 2 * d
    max_weight = 2 * (cutoff - 1)

    basis_2d, succ, pred, _hashes, _powers, _lookup = _get_cached_recurrence_data(
        ell, max_weight
    )

    G = np.zeros(len(basis_2d), dtype=np.complex128)
    G[0] = 1.0 + 0.0j
    _recurrence_fill_G(G, basis_2d, succ, pred, A, b, max_weight, ell)

    return G, basis_2d


class DensityMatrixCalculation(abc.ABC):
    _normalization: float

    def __init__(self, connector: BaseConnector):
        self.connector = connector
        # Cache for _get_density_matrix_element_from_recurrence (competitor API compat).
        self._density_matrix_element_cache: Dict[Tuple[int, ...], complex] = {}

    @abc.abstractmethod
    def calculate_hafnian(self, reduce_on: np.ndarray) -> float:
        """Calculates the hafnian given a reduction."""

    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the Bargmann A matrix and b vector.

        The default implementation reads ``self._A`` and ``self._linear``, matching
        the attribute names used in the competitor's test helpers and the concrete
        subclasses.  Subclasses may override this if they use different names.
        """
        return self._A, self._linear  # type: ignore[attr-defined]

    def _get_density_matrix_element_from_recurrence(
        self, reduce_on: Tuple[int, ...]
    ) -> complex:
        """Scalar recurrence for a single element (competitor-compatible API).

        This is provided for API compatibility with the competitor's interface.
        ``get_density_matrix`` uses the faster vectorised path instead.
        """
        if reduce_on in self._density_matrix_element_cache:
            return self._density_matrix_element_cache[reduce_on]
        k = np.array(reduce_on)
        i = int(np.flatnonzero(k)[0])
        previous = k.copy()
        previous[i] -= 1
        element = self._linear[i] * self._get_density_matrix_element_from_recurrence(  # type: ignore[attr-defined]
            tuple(previous)
        )
        for j, previous_j in enumerate(previous):
            if previous_j == 0:
                continue
            previous_previous = previous.copy()
            previous_previous[j] -= 1
            element += (
                np.sqrt(float(previous_j))
                * self._A[i, j]  # type: ignore[attr-defined]
                * self._get_density_matrix_element_from_recurrence(
                    tuple(previous_previous)
                )
            )
        element /= np.sqrt(float(k[i]))
        self._density_matrix_element_cache[reduce_on] = element
        return element

    def get_density_matrix_element(self, bra: np.ndarray, ket: np.ndarray) -> float:
        reduce_on = np.concatenate([ket, bra])

        return (
            self._normalization
            * self.calculate_hafnian(reduce_on)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(self, occupation_numbers: np.ndarray) -> np.ndarray:
        r"""Compute the full density matrix.

        For :class:`~piquasso._simulators.connectors.numpy_.NumpyConnector` this uses
        the fast recurrence relation from Eq. (45) of arXiv:2209.06069, which fills all
        Fock amplitudes in one compiled Numba pass and reuses cached index tables.

        For differentiable connectors (TensorFlow, JAX) the method falls back to the
        connector-compatible element-wise path so that gradients are preserved.

        Args:
            occupation_numbers: Array of shape ``(m, d)`` of Fock basis multi-indices.

        Returns:
            The complex ``(m, m)`` density matrix in the truncated Fock space.
        """
        if self.connector.allow_conditionals:
            return self._get_density_matrix_recurrence(occupation_numbers)

        # Differentiable fallback: use connector ops so gradients flow through.
        return self._get_density_matrix_elementwise(occupation_numbers)

    def _get_density_matrix_elementwise(
        self, occupation_numbers: np.ndarray
    ) -> np.ndarray:
        """Element-wise density matrix via hafnians. Differentiable."""
        n = occupation_numbers.shape[0]
        density_matrix = self.connector.np.empty(shape=(n, n), dtype=complex)
        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                density_matrix[i, j] = self.get_density_matrix_element(bra, ket)
        return density_matrix

    def _get_density_matrix_recurrence(
        self, occupation_numbers: np.ndarray
    ) -> np.ndarray:
        """Fast recurrence-based density matrix (NumpyConnector only)."""
        d = occupation_numbers.shape[1]
        cutoff = int(occupation_numbers.sum(axis=1).max()) + 1

        ell = 2 * d
        max_weight = 2 * (cutoff - 1)

        A, b = self._get_bargmann_Ab()

        # Retrieve cached basis + tables (built only once per (ell, max_weight)).
        basis_2d, succ, pred, hashes, powers, dense_lookup = (
            _get_cached_recurrence_data(ell, max_weight)
        )

        # Run the numba recurrence to fill all G values.
        G = np.zeros(len(basis_2d), dtype=np.complex128)
        G[0] = 1.0 + 0.0j
        _recurrence_fill_G(G, basis_2d, succ, pred, A, b, max_weight, ell)

        # Vectorised hash → flat-index lookup for all (ket ‖ bra) pairs.
        # powers[:d] correspond to the ket part, powers[d:] to the bra part.
        ket_powers = powers[:d]   # (d,)
        bra_powers = powers[d:]   # (d,)

        # (m,) hash contribution from each row of occupation_numbers as ket / bra.
        ket_hashes = occupation_numbers @ ket_powers   # (m,)
        bra_hashes = occupation_numbers @ bra_powers   # (m,)

        m = occupation_numbers.shape[0]
        density_matrix = np.empty((m, m), dtype=np.complex128)

        if dense_lookup is not None:
            # O(1) per element via the precomputed dense array.
            for i in range(m):
                # Upper triangle (including diagonal): ket = row j, bra = row i.
                combined = ket_hashes[i:] + bra_hashes[i]
                row_vals = self._normalization * G[dense_lookup[combined]]
                density_matrix[i, i:] = row_vals
                # Fill lower triangle by conjugation (density matrix is Hermitian).
                density_matrix[i + 1:, i] = row_vals[1:].conj()
        else:
            # Searchsorted fallback (large d / large cutoff).
            sort_idx = np.argsort(hashes)
            sorted_hashes = hashes[sort_idx]
            n_basis = len(hashes)

            for i in range(m):
                combined = ket_hashes[i:] + bra_hashes[i]
                pos = np.searchsorted(sorted_hashes, combined)
                pos_c = np.minimum(pos, n_basis - 1)
                flat_idx = sort_idx[pos_c]
                row_vals = self._normalization * G[flat_idx]
                density_matrix[i, i:] = row_vals
                density_matrix[i + 1:, i] = row_vals[1:].conj()

        return density_matrix

    def get_particle_number_detection_probabilities(
        self, occupation_numbers: np.ndarray
    ) -> np.ndarray:
        ret_list = []

        for occupation_number in occupation_numbers:
            probability = np.real(
                self.get_density_matrix_element(
                    bra=occupation_number,
                    ket=occupation_number,
                )
            )
            ret_list.append(probability)

        ret = np.array(ret_list, dtype=float)

        ret[abs(ret) < 1e-10] = 0.0

        return ret


class NondisplacedDensityMatrixCalculation(DensityMatrixCalculation):
    def __init__(
        self,
        complex_covariance: np.ndarray,
        connector: BaseConnector,
    ) -> None:
        super().__init__(connector=connector)

        np = connector.np

        d = len(complex_covariance) // 2
        Q = (complex_covariance + np.identity(2 * d)) / 2

        Qinv = np.linalg.inv(Q)
        identity = np.identity(d)
        zeros = np.zeros_like(identity)

        X = np.block(
            [
                [zeros, identity],
                [identity, zeros],
            ],
        )

        self._A = X @ (np.identity(2 * d, dtype=complex) - Qinv)
        self._linear = np.zeros(2 * d, dtype=complex)

        self._normalization: float = 1 / np.sqrt(np.linalg.det(Q))
        self._density_matrix_element_cache[(0,) * (2 * d)] = self._normalization

    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._linear

    def calculate_hafnian(self, reduce_on: np.ndarray) -> float:
        return self.connector.hafnian(self._A, reduce_on)


class DisplacedDensityMatrixCalculation(DensityMatrixCalculation):
    def __init__(
        self,
        complex_displacement: np.ndarray,
        complex_covariance: np.ndarray,
        connector: BaseConnector,
    ) -> None:
        super().__init__(connector=connector)

        np = connector.np

        d = len(complex_displacement) // 2
        Q = (complex_covariance + np.identity(2 * d)) / 2

        Qinv = np.linalg.inv(Q)
        identity = np.identity(d)
        zeros = np.zeros_like(identity)

        X = np.block(
            [
                [zeros, identity],
                [identity, zeros],
            ],
        )

        self._A = X @ (np.identity(2 * d, dtype=complex) - Qinv)

        self._gamma = complex_displacement.conj() @ Qinv
        self._linear = self._gamma

        self._normalization: float = np.exp(
            -0.5 * self._gamma @ complex_displacement
        ) / np.sqrt(np.linalg.det(Q))
        self._density_matrix_element_cache[(0,) * (2 * d)] = self._normalization

    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._linear

    def calculate_hafnian(self, reduce_on: np.ndarray) -> float:
        return self.connector.loop_hafnian(self._A, self._gamma, reduce_on)


def calculate_click_probability_nondisplaced(
    xpxp_covariance: np.ndarray,
    occupation_number: Tuple[int, ...],
) -> float:
    r"""
    Calculates the threshold detection probability with the equation

    .. math::
        p(S) = \frac{
            \operatorname{tor}( I - ( \Sigma^{-1} )_{(S)} )
        }{
            \sqrt{\operatorname{det}(\Sigma)}
        },

    where :math:`\Sigma \in \mathbb{R}^{2d \times 2d}` is a symmetric matrix
    defined by

    .. math::
        \Sigma = \frac{1}{2} \left (
                \frac{1}{\hbar} \sigma_{xp}
                + I
            \right ).
    """

    d = len(xpxp_covariance) // 2

    sigma: np.ndarray = (xpxp_covariance + np.identity(2 * d)) / 2

    sigma_inv = np.linalg.inv(sigma)

    sigma_inv_reduced = block_reduce_xpxp(sigma_inv, reduce_on=occupation_number)

    A = np.identity(len(sigma_inv_reduced), dtype=float) - sigma_inv_reduced

    probability = torontonian(A) / np.sqrt(np.linalg.det(sigma))

    return max(probability, 0.0)


def calculate_click_probability(
    xpxp_covariance: np.ndarray,
    xpxp_mean: np.ndarray,
    occupation_number: Tuple[int, ...],
) -> float:
    r"""
    Calculates the threshold detection probability with the equation

    .. math::
        p(S) = \frac{
            \operatorname{ltor}( I - ( \Sigma^{-1} )_{(S)}, \vec{\gamma} )
        }{
            \sqrt{\operatorname{det}(\Sigma)}
        },

    where :math:`\Sigma \in \mathbb{R}^{2d \times 2d}` is a symmetric matrix
    defined by

    .. math::
        \Sigma = \frac{1}{2} \left (
                \frac{1}{\hbar} \sigma_{xp}
                + I
            \right ),

    and

    .. math::
        \vec{\gamma} = \Sigma^{-1} \vec{\alpha}
    """

    d = len(xpxp_covariance) // 2

    sigma: np.ndarray = (xpxp_covariance + np.identity(2 * d)) / 2

    sigma_inv = np.linalg.inv(sigma)

    gamma = sigma_inv @ xpxp_mean

    sigma_inv_reduced = block_reduce_xpxp(sigma_inv, reduce_on=occupation_number)

    gamma_reduced = block_reduce_xpxp(gamma, reduce_on=occupation_number)

    A = np.identity(len(sigma_inv_reduced), dtype=float) - sigma_inv_reduced

    exponential_term = np.exp(-xpxp_mean @ sigma_inv @ xpxp_mean / 2)

    probability = (
        loop_torontonian(A, gamma_reduced).real
        * exponential_term
        / np.sqrt(np.linalg.det(sigma))
    )

    return max(probability, 0.0)