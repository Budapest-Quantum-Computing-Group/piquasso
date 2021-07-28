#
# Copyright 2021 Budapest Quantum Computing Group
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

import numpy as np

from scipy.special import factorial

from piquasso import constants
from piquasso._math.linalg import block_reduce, reduce_
from piquasso._math.hafnian import loop_hafnian
from piquasso._math.torontonian import torontonian


class DensityMatrixCalculation:
    def __init__(self, complex_displacement, complex_covariance) -> None:
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

        self._normalization = (
            np.exp(-0.5 * self._gamma @ complex_displacement)
            / np.sqrt(np.linalg.det(Q))
        )

    def _get_A_reduced(self, reduce_on: tuple):
        A_reduced = reduce_(self._A, reduce_on=reduce_on)

        np.fill_diagonal(
            A_reduced,
            reduce_(
                self._gamma, reduce_on=reduce_on
            )
        )

        return A_reduced

    def get_density_matrix_element(self, bra: tuple, ket: tuple) -> float:
        reduce_on = ket + bra

        A_reduced = self._get_A_reduced(reduce_on=reduce_on)

        return (
            self._normalization * loop_hafnian(A_reduced)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(self, occupation_numbers):
        n = len(occupation_numbers)

        density_matrix = np.empty(shape=(n, n), dtype=complex)

        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                density_matrix[i, j] = self.get_density_matrix_element(bra, ket)

        return density_matrix

    def get_particle_number_detection_probabilities(
        self,
        occupation_numbers: list
    ) -> list:
        ret = []

        for occupation_number in occupation_numbers:
            probability = np.real(
                self.get_density_matrix_element(
                    bra=occupation_number,
                    ket=occupation_number,
                )
            )
            ret.append(probability)

        ret = np.array(ret, dtype=float)

        ret[abs(ret) < 1e-10] = 0.0

        return ret


class ThresholdCalculation:
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

    def __init__(self, xp_covariance) -> None:
        d = len(xp_covariance) // 2

        self._sigma = (xp_covariance / constants.HBAR + np.identity(2 * d)) / 2

        self._normalization = 1 / np.sqrt(np.linalg.det(self._sigma))

    def _get_sigma_inv_reduced(self, reduce_on: tuple):
        return (
            block_reduce(
                np.linalg.inv(self._sigma),
                reduce_on=reduce_on,
            )
        )

    def calculate_click_probability(self, occupation_number):
        sigma_inv_reduced = self._get_sigma_inv_reduced(reduce_on=occupation_number)

        return self._normalization * torontonian(
            np.identity(len(sigma_inv_reduced)) - sigma_inv_reduced
        )
