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
from piquasso._math.linalg import block_reduce
from piquasso._math.hafnian import loop_hafnian
from piquasso._math.torontonian import torontonian


def calculate_particle_number_detection_probability(
    state,
    subspace_modes: tuple,
    occupation_numbers: tuple,
):
    d = len(subspace_modes)
    Q = (state.complex_covariance + np.identity(2 * d)) / 2
    Qinv = np.linalg.inv(Q)

    identity = np.identity(d)
    zeros = np.zeros_like(identity)

    X = np.block(
        [
            [zeros, identity],
            [identity, zeros],
        ],
    )

    A = X @ (np.identity(2 * d, dtype=complex) - Qinv)

    alpha = state.complex_displacement
    gamma = alpha.conj() @ Qinv

    A_reduced = block_reduce(A, reduce_on=occupation_numbers)

    np.fill_diagonal(
        A_reduced,
        block_reduce(
            gamma, reduce_on=occupation_numbers
        )
    )

    return (
        loop_hafnian(A_reduced) * np.exp(-0.5 * gamma @ alpha)
        / (np.prod(factorial(occupation_numbers)) * np.sqrt(np.linalg.det(Q)))
    ).real


def calculate_threshold_detection_probability(
    state,
    subspace_modes,
    occupation_numbers,
):
    r"""
    Calculates the threshold detection probability with the equation

    .. math::
        p(S) = \frac{
            \operatorname{tor}( I - ( \Sigma^{-1} )_{(S)} )
        }{
            \sqrt{\operatorname{det}(\Sigma)}
        },

    where :math:`\Sigma \in \mathbb{R}^{2d \times 2d}` is a symmetric matrix defined by

    .. math::
        \Sigma = \frac{1}{2} \left (
                \frac{1}{\hbar} \sigma_{xp}
                + I
            \right ).
    """

    d = len(subspace_modes)

    sigma = (state.xp_cov / constants.HBAR + np.identity(2 * d)) / 2

    sigma_inv_reduced = (
        block_reduce(
            np.linalg.inv(sigma),
            reduce_on=occupation_numbers,
        )
    )

    return torontonian(
        np.identity(len(sigma_inv_reduced)) - sigma_inv_reduced
    ) / np.sqrt(np.linalg.det(sigma))
