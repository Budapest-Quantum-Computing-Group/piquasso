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

from typing import Tuple

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


def _build_index_tables(
    basis_2d: np.ndarray,
    max_weight: int,
    ell: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute successor and predecessor flat-index tables for the recurrence.

    Args:
        basis_2d: Array of shape (N, ell) of multi-indices sorted by weight.
        max_weight: Maximum total photon number (= 2*(cutoff-1)).
        ell: Dimension of the multi-index.

    Returns:
        succ: int64 array of shape (N, ell); succ[idx, i] is the flat index of
              basis[idx] + e_i, or -1 if that index exceeds max_weight.
        pred: int64 array of shape (N, ell); pred[idx, j] is the flat index of
              basis[idx] - e_j, or -1 if basis[idx][j] == 0.
    """
    n = len(basis_2d)
    # Build a lookup dict: tuple(k) -> flat_idx
    lookup = {tuple(basis_2d[idx].tolist()): idx for idx in range(n)}

    succ = np.full((n, ell), -1, dtype=np.int64)
    pred = np.full((n, ell), -1, dtype=np.int64)

    for idx in range(n):
        k = basis_2d[idx]
        w = int(k.sum())
        for i in range(ell):
            # Successor
            if w < max_weight:
                k_new = k.copy()
                k_new[i] += 1
                t = tuple(k_new.tolist())
                if t in lookup:
                    succ[idx, i] = lookup[t]
            # Predecessor
            if k[i] > 0:
                k_prev = k.copy()
                k_prev[i] -= 1
                t = tuple(k_prev.tolist())
                if t in lookup:
                    pred[idx, i] = lookup[t]

    return succ, pred


def _compute_G_array_via_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    cutoff: int,
    d: int,
) -> np.ndarray:
    r"""Compute the full array of renormalized Fock amplitudes using the recurrence
    relation from Eq. (45) of arXiv:2209.06069.

    For a Gaussian state with Bargmann triple (A, b, c), the density matrix element is

    .. math::
        \langle m | \rho | n \rangle = c \cdot \tilde{G}_{n \oplus m},

    where :math:`\tilde{G}_{k} = G^A_k(b) / \sqrt{k!}` is the renormalized
    multidimensional Hermite polynomial, computed via the recurrence starting from
    :math:`\tilde{G}_0 = 1`.

    Args:
        A: The 2d x 2d Bargmann A matrix.
        b: The 2d Bargmann b vector.
        cutoff: Fock space truncation cutoff.
        d: Number of modes.

    Returns:
        numpy.ndarray: 1D complex array G of renormalized Fock amplitudes, indexed by
        the position of the 2d-mode multi-index in the basis sorted by weight
        (k = ket ⊕ bra).
    """
    ell = 2 * d
    max_weight = 2 * (cutoff - 1)

    basis_2d = nb_get_fock_space_basis(d=ell, cutoff=max_weight + 1)
    succ, pred = _build_index_tables(basis_2d, max_weight, ell)

    G = np.zeros(len(basis_2d), dtype=np.complex128)
    G[0] = 1.0 + 0.0j

    _recurrence_fill_G(G, basis_2d, succ, pred, A, b, max_weight, ell)

    return G, basis_2d


class DensityMatrixCalculation(abc.ABC):
    _normalization: float

    def __init__(self, connector: BaseConnector):
        self.connector = connector

    @abc.abstractmethod
    def calculate_hafnian(self, reduce_on: np.ndarray) -> float:
        """Calculates the hafnian given a reduction."""

    @abc.abstractmethod
    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the Bargmann A matrix and b vector for this state."""

    def get_density_matrix_element(self, bra: np.ndarray, ket: np.ndarray) -> float:
        reduce_on = np.concatenate([ket, bra])

        return (
            self._normalization
            * self.calculate_hafnian(reduce_on)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(self, occupation_numbers: np.ndarray) -> np.ndarray:
        r"""Compute the full density matrix using the recurrence relation from
        Eq. (45) of arXiv:2209.06069 (Yao, Miatto, Quesada, 2022).

        This method is significantly faster than computing each matrix element
        independently via hafnians, because it reuses previously computed
        Fock amplitudes via a linear recurrence on the multidimensional Hermite
        polynomials that represent the Gaussian state in Fock space.

        The density matrix element is given by:

        .. math::
            \langle m | \rho | n \rangle = c \cdot \tilde{G}_{n \oplus m},

        where :math:`c` is the normalization scalar, and :math:`\tilde{G}_{k}` are the
        renormalized multidimensional Hermite polynomial values computed recursively.

        Args:
            occupation_numbers (numpy.ndarray):
                Array of multi-indices representing the Fock space basis.

        Returns:
            numpy.ndarray: The complex density matrix in the truncated Fock space.
        """
        d = occupation_numbers.shape[1]
        cutoff = int(occupation_numbers.sum(axis=1).max()) + 1

        A, b = self._get_bargmann_Ab()
        G, basis_2d = _compute_G_array_via_recurrence(A, b, cutoff, d)

        # Build a lookup from multi-index tuple -> position in basis_2d (= flat G index).
        # This avoids calling get_index_in_fock_space_array in the inner loop.
        lookup = {tuple(basis_2d[idx].tolist()): idx for idx in range(len(basis_2d))}

        n = occupation_numbers.shape[0]
        density_matrix = np.empty(shape=(n, n), dtype=complex)

        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                # k = ket ⊕ bra (matches reduce_on = [ket, bra] convention)
                k = tuple(ket.tolist() + bra.tolist())
                density_matrix[i, j] = self._normalization * G[lookup[k]]

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
        self._b = np.zeros(2 * d, dtype=complex)

        self._normalization: float = 1 / np.sqrt(np.linalg.det(Q))

    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._b

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
        # b vector for the recurrence: b = gamma = mu* @ Q^{-1}
        self._b = self._gamma

        self._normalization: float = np.exp(
            -0.5 * self._gamma @ complex_displacement
        ) / np.sqrt(np.linalg.det(Q))

    def _get_bargmann_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._b

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
