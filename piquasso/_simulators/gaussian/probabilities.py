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
from typing import Dict, Tuple

import numpy as np
import numba as nb

from scipy.special import factorial

from piquasso._math.linalg import block_reduce_xpxp
from piquasso._math.torontonian import torontonian, loop_torontonian
from piquasso.api.connector import BaseConnector



@nb.njit(cache=True)
def _recurrence_fill_G(
    G: np.ndarray,
    basis_2d: np.ndarray,
    succ: np.ndarray,
    pred: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
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

    Boundary handling is entirely delegated to the ``succ`` table: any successor
    that would fall outside the valid basis has ``succ[idx, i] == -1``, so no
    separate weight check is required inside this kernel.

    Args:
        G: Flat 1D array of complex Fock amplitudes indexed by Fock space position.
           Must be pre-allocated with G[0] = 1.0 and all other entries 0.
        basis_2d: Array of shape (N, ell) of multi-indices, sorted by total weight
                  (ascending) so that all predecessors are processed before any
                  successor.
        succ: Precomputed successor table of shape (N, ell).
              succ[idx, i] = flat index of basis[idx] + e_i, or -1 if out of range.
        pred: Precomputed predecessor table of shape (N, ell).
              pred[idx, j] = flat index of basis[idx] - e_j, or -1 if k[j] == 0.
        A: The ell x ell complex symmetric Bargmann A matrix.
        b: The ell-dimensional complex Bargmann b vector.
        ell: Dimension of the multi-index (= 2*d for d modes).
    """
    for idx in range(len(basis_2d)):
        k = basis_2d[idx]
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
                    val += np.sqrt(np.float64(kj)) * A[i, j] * G[pred[idx, j]]

            G[new_flat_idx] = val / np.sqrt(np.float64(ki + 1))


@lru_cache(maxsize=64)
def _get_cached_recurrence_data(
    d: int,
    cutoff: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build and cache the Fock-space basis, successor/predecessor tables, and a
    flat-index matrix for computing the full density matrix via the recurrence
    from Eq. (45) of arXiv:2209.06069.

    Unlike the simplex-basis approach (which builds all ``2d``-dimensional
    multi-indices with total weight ``≤ 2*(cutoff-1)``), this function builds the
    **cartesian-product** basis: every pair ``(n, m)`` where both ``n`` and ``m``
    are valid ``d``-mode Fock states (i.e. each has total photon number
    ``≤ cutoff-1``).  The cartesian basis is ``~2–3×`` smaller than the simplex
    basis, so the Numba recurrence kernel does proportionally less work.

    The ``flat_index_matrix[i, j]`` entry stores the position of the pair
    ``(ket=occ[j], bra=occ[i])`` inside the weight-sorted cartesian basis, where
    ``occ = get_fock_space_basis(d=d, cutoff=cutoff)``.  After the recurrence has
    filled ``G``, the full density matrix is recovered as::

        density_matrix = normalization * G[flat_index_matrix]

    in a single vectorised NumPy indexing call — no Python loop over basis rows.

    Successor/predecessor tables are built with the same mixed-radix hash scheme
    as before, but with per-half validity guards:

    * For ket dimensions (``i < d``): the successor is valid only when
      ``ket_weight < cutoff - 1``.
    * For bra dimensions (``i ≥ d``): the successor is valid only when
      ``bra_weight < cutoff - 1``.

    This replaces the previous single ``total_weight < max_weight`` guard and
    ensures that the successor always lands on another element of the cartesian
    basis.

    Args:
        d: Number of modes.
        cutoff: Fock-space cutoff (maximum total photon number per half + 1).

    Returns:
        cart_basis:        ``(m², ell)`` int64 array of 2d-dim multi-indices
                           sorted by total weight (ascending), where ``ell = 2*d``
                           and ``m = len(get_fock_space_basis(d, cutoff))``.
        succ:              ``(m², ell)`` int64 successor table (``-1`` if the
                           successor would leave the cartesian basis).
        pred:              ``(m², ell)`` int64 predecessor table (``-1`` if
                           ``k[j] == 0``).
        flat_index_matrix: ``(m, m)`` int64 array where entry ``[i, j]`` is the
                           flat index (row in ``cart_basis``) of the pair
                           ``(ket=occ[j], bra=occ[i])``.
    """
    from piquasso._math.fock import get_fock_space_basis as _gfsb

    occ: np.ndarray = _gfsb(d=d, cutoff=cutoff)   # (m, d)
    m: int = len(occ)
    ell: int = 2 * d

    # --- Build cartesian basis sorted by total weight ---
    # cart_basis_unsorted[i*m + j] = (occ[i] as ket-half, occ[j] as bra-half)
    ket_rep: np.ndarray = np.repeat(occ, m, axis=0)   # (m², d)
    bra_rep: np.ndarray = np.tile(occ, (m, 1))         # (m², d)
    cart_basis_unsorted: np.ndarray = np.hstack([ket_rep, bra_rep])  # (m², ell)

    weights: np.ndarray = cart_basis_unsorted.sum(axis=1)
    order: np.ndarray = np.argsort(weights, kind="stable")
    cart_basis: np.ndarray = cart_basis_unsorted[order]  # (m², ell), sorted by weight

    # inv_order[i*m + j] = position of (occ[i], occ[j]) in cart_basis.
    # flat_index_matrix[bra_i, ket_j] = position of (ket=occ[ket_j], bra=occ[bra_i])
    #                                  = inv_order[ket_j * m + bra_i]
    # Equivalently: flat_index_matrix = inv_order.reshape(m, m).T
    inv_order: np.ndarray = np.empty(m * m, dtype=np.int64)
    inv_order[order] = np.arange(m * m, dtype=np.int64)
    flat_index_matrix: np.ndarray = inv_order.reshape(m, m).T  # (m, m)

    # --- Build successor / predecessor tables via mixed-radix hashing ---
    N: int = m * m
    max_weight: int = 2 * (cutoff - 1)
    radix: int = max_weight + 2
    powers: np.ndarray = radix ** np.arange(ell, dtype=np.int64)
    hashes: np.ndarray = cart_basis @ powers   # (m²,)

    max_hash: int = int(hashes.max())
    succ: np.ndarray = np.full((N, ell), -1, dtype=np.int64)
    pred: np.ndarray = np.full((N, ell), -1, dtype=np.int64)

    # Per-half photon-number sums for validity guard.
    ket_weights: np.ndarray = cart_basis[:, :d].sum(axis=1)  # (m²,)
    bra_weights: np.ndarray = cart_basis[:, d:].sum(axis=1)  # (m²,)

    if max_hash < 50_000_000:
        # Dense O(1) lookup – fits comfortably in memory.
        dense_lookup: np.ndarray = np.full(max_hash + 1, -1, dtype=np.int64)
        dense_lookup[hashes] = np.arange(N, dtype=np.int64)

        for i in range(ell):
            new_h = hashes + powers[i]
            # Guard: the successor must keep the relevant half within [0, cutoff-1].
            half_weights = ket_weights if i < d else bra_weights
            valid = (half_weights < cutoff - 1) & (new_h <= max_hash)
            rows = np.where(valid)[0]
            cand = dense_lookup[new_h[rows]]
            good = cand >= 0
            succ[rows[good], i] = cand[good]

            # Predecessors: subtract powers[i], valid only when k[i] > 0.
            rows2 = np.where(cart_basis[:, i] > 0)[0]
            pred[rows2, i] = dense_lookup[hashes[rows2] - powers[i]]
    else:
        # Searchsorted fallback for very large cutoffs.
        sort_idx = np.argsort(hashes)
        sorted_hashes = hashes[sort_idx]

        for i in range(ell):
            new_h = hashes + powers[i]
            half_weights = ket_weights if i < d else bra_weights
            mask = half_weights < cutoff - 1
            pos = np.searchsorted(sorted_hashes, new_h)
            pos_c = np.minimum(pos, N - 1)
            found = mask & (pos < N) & (sorted_hashes[pos_c] == new_h)
            succ[found, i] = sort_idx[pos[found]]

            rows2 = np.where(cart_basis[:, i] > 0)[0]
            new_h2 = hashes[rows2] - powers[i]
            pos2 = np.searchsorted(sorted_hashes, new_h2)
            pos2_c = np.minimum(pos2, N - 1)
            found2 = sorted_hashes[pos2_c] == new_h2
            pred[rows2[found2], i] = sort_idx[pos2[found2]]

    return cart_basis, succ, pred, flat_index_matrix



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
        r"""Fast recurrence-based density matrix (NumpyConnector only).

        Fills all ``m²`` Fock amplitudes via a single Numba pass over the
        cartesian-product basis (see :func:`_get_cached_recurrence_data`), then
        extracts the full density matrix with one vectorised NumPy indexing call::

            density_matrix = normalization * G[flat_index_matrix]

        No Python loop over basis rows is needed at extraction time.

        Args:
            occupation_numbers: Array of shape ``(m, d)`` returned by
                :func:`~piquasso._math.fock.get_fock_space_basis`.

        Returns:
            Complex ``(m, m)`` density matrix.
        """
        d = occupation_numbers.shape[1]
        cutoff = int(occupation_numbers.sum(axis=1).max()) + 1
        ell = 2 * d

        A, b = self._get_bargmann_Ab()

        # Retrieve cached cartesian basis + tables (built once per (d, cutoff)).
        cart_basis, succ, pred, flat_index_matrix = (
            _get_cached_recurrence_data(d, cutoff)
        )

        # Run the Numba recurrence over the cartesian basis.
        G = np.zeros(len(cart_basis), dtype=np.complex128)
        G[0] = 1.0 + 0.0j
        _recurrence_fill_G(G, cart_basis, succ, pred, A, b, ell)

        # Single vectorised extraction: flat_index_matrix[i, j] holds the flat
        # index of (ket=occ[j], bra=occ[i]) inside cart_basis, so
        # G[flat_index_matrix] gives the full (m, m) array of renormalised
        # Hermite-polynomial values in one shot.
        return self._normalization * G[flat_index_matrix]

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