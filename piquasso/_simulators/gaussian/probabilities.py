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

import abc

from typing import Tuple

import numpy as np

from scipy.special import factorial

from piquasso._math.linalg import block_reduce_xpxp
from piquasso._math.torontonian import torontonian, loop_torontonian
from piquasso.api.connector import BaseConnector


class DensityMatrixCalculation(abc.ABC):
    _normalization: float

    def __init__(self, connector: BaseConnector):
        self.connector = connector

    @abc.abstractmethod
    def calculate_hafnian(self, reduce_on: np.ndarray) -> float:
        """Calculates the hafnian given a reduction."""

    def get_density_matrix_element(self, bra: np.ndarray, ket: np.ndarray) -> float:
        reduce_on = np.concatenate([ket, bra])

        return (
            self._normalization
            * self.calculate_hafnian(reduce_on)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(self, occupation_numbers: np.ndarray) -> np.ndarray:
        n = occupation_numbers.shape[0]

        density_matrix = np.empty(shape=(n, n), dtype=complex)

        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                density_matrix[i, j] = self.get_density_matrix_element(bra, ket)

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

        self._normalization: float = 1 / np.sqrt(np.linalg.det(Q))

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

        self._normalization: float = np.exp(
            -0.5 * self._gamma @ complex_displacement
        ) / np.sqrt(np.linalg.det(Q))

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
