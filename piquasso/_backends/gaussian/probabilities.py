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
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from scipy.special import factorial

from piquasso import constants
from piquasso._math.linalg import block_reduce, reduce_
from piquasso._math.hafnian import loop_hafnian
from piquasso._math.torontonian import torontonian


class DensityMatrixCalculation:
    def __init__(
        self,
        complex_displacement: npt.NDArray[np.complex128],
        complex_covariance: npt.NDArray[np.complex128],
    ) -> None:
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

        self._A: npt.NDArray[np.complex128] = \
            X @ (np.identity(2 * d, dtype=complex) - Qinv)

        self._gamma: npt.NDArray[np.complex128] = \
            complex_displacement.conj() @ Qinv

        self._normalization: npt.NDArray[np.complex128] = (
            np.exp(-0.5 * self._gamma @ complex_displacement)
            / np.sqrt(np.linalg.det(Q))
        )

    def _get_A_reduced(self, reduce_on: Tuple[int, ...]) -> npt.NDArray[np.complex128]:
        A_reduced = reduce_(self._A, reduce_on=reduce_on)

        np.fill_diagonal(
            A_reduced,
            reduce_(
                self._gamma, reduce_on=reduce_on
            )
        )

        return A_reduced

    def get_density_matrix_element(
        self, bra: Tuple[int, ...], ket: Tuple[int, ...]
    ) -> float:
        reduce_on = ket + bra

        A_reduced = self._get_A_reduced(reduce_on=reduce_on)

        return (
            self._normalization * loop_hafnian(A_reduced)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(
        self, occupation_numbers: List[Tuple[int, ...]]
    ) -> npt.NDArray[np.complex128]:
        n = len(occupation_numbers)

        density_matrix = np.empty(shape=(n, n), dtype=complex)

        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                density_matrix[i, j] = self.get_density_matrix_element(bra, ket)

        return density_matrix

    def get_particle_number_detection_probabilities(
        self,
        occupation_numbers: List[Tuple[int, ...]]
    ) -> npt.NDArray[np.float64]:
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

    def __init__(self, xp_covariance: npt.NDArray[np.float64]) -> None:
        d = len(xp_covariance) // 2

        self._sigma: npt.NDArray[np.float64] = \
            (xp_covariance / constants.HBAR + np.identity(2 * d)) / 2

        self._normalization = 1 / np.sqrt(np.linalg.det(self._sigma))

    def _get_sigma_inv_reduced(
        self, reduce_on: Tuple[int, ...]
    ) -> npt.NDArray[np.float64]:
        return (
            block_reduce(
                np.linalg.inv(self._sigma),
                reduce_on=reduce_on,
            )
        )

    def calculate_click_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        sigma_inv_reduced = self._get_sigma_inv_reduced(reduce_on=occupation_number)

        return self._normalization * (torontonian(
            np.identity(len(sigma_inv_reduced), dtype=float) - sigma_inv_reduced
        )).real
