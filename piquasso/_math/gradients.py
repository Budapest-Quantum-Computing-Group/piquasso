#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from typing import Callable

from piquasso.api.calculator import BaseCalculator


def create_single_mode_displacement_gradient(
    r: float,
    phi: float,
    cutoff: int,
    transformation: np.ndarray,
    calculator: BaseCalculator,
) -> Callable:
    def grad(upstream):
        np = calculator.fallback_np
        tf = calculator._tf

        epiphi = np.exp(1j * phi)
        eimphi = np.exp(-1j * phi)

        r_grad = np.zeros((cutoff,) * 2, dtype=complex)
        phi_grad = np.zeros((cutoff,) * 2, dtype=complex)
        # NOTE: This algorithm deliberately overindexes the gate matrix.
        for row in range(cutoff):
            for col in range(cutoff):
                r_grad[row, col] = (
                    -r * transformation[row, col]
                    + epiphi * np.sqrt(row) * transformation[row - 1, col]
                    - eimphi * np.sqrt(col) * transformation[row, col - 1]
                )
                phi_grad[row, col] = (
                    r
                    * 1j
                    * (
                        np.sqrt(row) * epiphi * transformation[row - 1, col]
                        + np.sqrt(col) * eimphi * transformation[row, col - 1]
                    )
                )
        r_grad_sum = tf.math.real(tf.reduce_sum(upstream * r_grad))
        phi_grad_sum = tf.math.real(tf.reduce_sum(upstream * tf.math.conj(phi_grad)))
        return (r_grad_sum, phi_grad_sum)

    return grad


def create_single_mode_squeezing_gradient(
    r: float,
    phi: float,
    cutoff: int,
    transformation: np.ndarray,
    calculator: BaseCalculator,
) -> Callable:
    def grad(upstream):
        np = calculator.fallback_np
        tf = calculator._tf

        r_grad = np.zeros((cutoff,) * 2, dtype=complex)
        phi_grad = np.zeros((cutoff,) * 2, dtype=complex)
        sinhr = np.sinh(r)
        coshr = np.cosh(r)
        sechr = 1 / coshr
        tanhr = np.tanh(r)
        c_coeff = -sinhr / (np.sqrt(2 * np.power(coshr, 3)))
        sum_coeff = -(sechr**2) / 2
        eiphi = np.exp(1j * phi)
        emiphi = np.exp(-1j * phi)

        # NOTE: This algorithm deliberately overindexes the gate matrix.
        for row in range(cutoff):
            for col in range(cutoff):
                r_grad[row, col] = (
                    c_coeff * transformation[row, col]
                    - np.sqrt(row * col)
                    * sechr
                    * tanhr
                    * transformation[row - 1, col - 1]
                    + sum_coeff
                    * (np.sqrt(row * (row - 1)) * eiphi * transformation[row - 2, col])
                    + np.sqrt(col * (col - 1)) * emiphi * transformation[row, col - 2]
                )
                phi_grad[row, col] = (
                    -tanhr
                    * 1j
                    * (
                        np.sqrt(row * (row - 1)) * eiphi * transformation[row - 2, col]
                        + np.sqrt(col * (col - 1))
                        * emiphi
                        * transformation[row, col - 2]
                    )
                    / 2
                )

        # NOTE: Possibly Tensorflow bug, cast needed.
        # cannot compute AddN as input #1(zero-based) was expected to be\
        #  a double tensor but is a float tensor [Op:AddN].
        # The bug does not occur with Displacement gradient for unknown reasons.
        r_grad_sum = tf.cast(tf.math.real(tf.reduce_sum(upstream * r_grad)), tf.float32)
        phi_grad_sum = tf.math.real(tf.reduce_sum(upstream * tf.math.conj(phi_grad)))

        return (r_grad_sum, phi_grad_sum)

    return grad
