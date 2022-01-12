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

from scipy.special import factorial

from piquasso._math.linalg import block_reduce, reduce_
from piquasso._math.torontonian import torontonian

from piquasso.api.typing import HafnianFunction


class DensityMatrixCalculation:
    def __init__(
        self,
        complex_displacement: np.ndarray,
        complex_covariance: np.ndarray,
        loop_hafnian_function: HafnianFunction,
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

        self._A: np.ndarray = X @ (np.identity(2 * d, dtype=complex) - Qinv)

        self._gamma: np.ndarray = complex_displacement.conj() @ Qinv

        self._normalization: np.ndarray = np.exp(
            -0.5 * self._gamma @ complex_displacement
        ) / np.sqrt(np.linalg.det(Q))

        self.loop_hafnian_function = loop_hafnian_function

    def _get_A_reduced(self, reduce_on: Tuple[int, ...]) -> np.ndarray:
        A_reduced = reduce_(self._A, reduce_on=reduce_on)

        np.fill_diagonal(A_reduced, reduce_(self._gamma, reduce_on=reduce_on))

        return A_reduced

    def get_density_matrix_element(
        self, bra: Tuple[int, ...], ket: Tuple[int, ...]
    ) -> float:
        reduce_on = ket + bra

        A_reduced = self._get_A_reduced(reduce_on=reduce_on)

        return (
            self._normalization
            * self.loop_hafnian_function(A_reduced)
            / np.sqrt(np.prod(factorial(reduce_on)))
        )

    def get_density_matrix(
        self, occupation_numbers: List[Tuple[int, ...]]
    ) -> np.ndarray:
        n = len(occupation_numbers)

        density_matrix = np.empty(shape=(n, n), dtype=complex)

        for i, bra in enumerate(occupation_numbers):
            for j, ket in enumerate(occupation_numbers):
                density_matrix[i, j] = self.get_density_matrix_element(bra, ket)

        return density_matrix

    def get_particle_number_detection_probabilities(
        self, occupation_numbers: List[Tuple[int, ...]]
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


def calculate_click_probability(
    xp_covariance: np.ndarray,
    occupation_number: Tuple[int, ...],
    hbar: float,
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

    d = len(xp_covariance) // 2

    sigma: np.ndarray = (xp_covariance / hbar + np.identity(2 * d)) / 2

    sigma_inv_reduced = block_reduce(
        np.linalg.inv(sigma),
        reduce_on=occupation_number,
    )

    probability = (
        torontonian(
            np.identity(len(sigma_inv_reduced), dtype=float) - sigma_inv_reduced
        )
    ).real / np.sqrt(np.linalg.det(sigma))

    return max(probability, 0.0)
