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
    def displacement_matrix_gradient(upstream):
        np = calculator.fallback_np

        index_sqrts = np.sqrt(range(cutoff))

        row_sqrts = index_sqrts * np.exp(1j * phi)
        col_sqrts = index_sqrts * np.exp(-1j * phi)

        row_rolled_transformation = np.roll(transformation, 1, axis=0)
        col_rolled_transformation = np.roll(transformation, 1, axis=1)

        # NOTE: This algorithm rolls the last elements of the transormation matrix to
        # the 0th place, but the 0th element of `row_sqrts` and `col_sqrts` is always
        # zero, so it is fine.
        phi_grad = (
            row_sqrts * row_rolled_transformation.T
        ).T + col_sqrts * col_rolled_transformation

        r_grad = -r * transformation + (row_sqrts * row_rolled_transformation.T).T - col_sqrts * col_rolled_transformation

        phi_grad *= r * 1j

        tf = calculator._tf
        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            upstream = upstream.numpy()
            r_grad_sum = tf.constant(np.real(np.sum(upstream * np.conj(r_grad))))
            phi_grad_sum = tf.constant(np.real(np.sum(upstream * np.conj(phi_grad))))
        else:
            r_grad_sum = tf.math.real(tf.reduce_sum(upstream * tf.math.conj(r_grad)))
            phi_grad_sum = tf.math.real(
                tf.reduce_sum(upstream * tf.math.conj(phi_grad))
            )

        return (r_grad_sum, phi_grad_sum)

    return displacement_matrix_gradient


def create_single_mode_squeezing_gradient(
    r: float,
    phi: float,
    cutoff: int,
    transformation: np.ndarray,
    calculator: BaseCalculator,
) -> Callable:
    def squeezing_matrix_gradient(upstream):
        np = calculator.fallback_np

        sechr = 1 / np.cosh(r)
        tanhr = np.tanh(r)

        index_sqrts = np.sqrt(np.arange(cutoff, dtype=np.complex128))

        falling_index_sqrts = index_sqrts * np.roll(index_sqrts, 1)

        row_sqrts = falling_index_sqrts * np.exp(1j * phi)
        col_sqrts = falling_index_sqrts * np.exp(-1j * phi)

        row_rolled_transformation = np.roll(transformation, 2, axis=0)
        col_rolled_transformation = np.roll(transformation, 2, axis=1)

        # NOTE: This algorithm rolls the last and penultimate elements of the
        # transormation matrix to the 1st and 0th place, but the 0th and 1st element of
        # `row_sqrts` and `col_sqrts` is always zero, so it is fine.
        phi_grad = -0.5j * tanhr * ((
            row_sqrts * row_rolled_transformation.T
        ).T + col_sqrts * col_rolled_transformation)

        diagonally_rolled_transformation = np.roll(transformation, (1, 1), axis=(0, 1))

        r_grad = (
            (-tanhr * 0.5) * transformation
            - (sechr * tanhr) * np.outer(index_sqrts, index_sqrts) * diagonally_rolled_transformation
            - (sechr**2 * 0.5) * (row_sqrts * row_rolled_transformation.T).T
            + (sechr**2 * 0.5) * (col_sqrts * col_rolled_transformation)
            )
        """
        T = transformation
        grad_r = np.zeros((cutoff, cutoff), dtype=np.complex128)
        grad_phi = np.zeros((cutoff, cutoff), dtype=np.complex128)

        eiphi = np.exp(1j * phi)
        eiphiconj = np.exp(-1j * phi)

        for m in range(cutoff):
            for n in range(cutoff):
                grad_r[m, n] = (
                    -0.5 * tanhr * T[m, n]
                    - sechr * tanhr * index_sqrts[m] * index_sqrts[n] * T[m - 1, n - 1]
                    - 0.5 * eiphi * sechr**2 * index_sqrts[m] * index_sqrts[m - 1] * T[m - 2, n]
                    + 0.5 * eiphiconj * sechr**2 * index_sqrts[n] * index_sqrts[n - 1] * T[m, n - 2]
                )
                grad_phi[m, n] = (
                    -0.5j * eiphi * tanhr * index_sqrts[m] * index_sqrts[m - 1] * T[m - 2, n]
                    - 0.5j * eiphiconj * tanhr * index_sqrts[n] * index_sqrts[n - 1] * T[m, n - 2]
                )
        """
        tf = calculator._tf

        static_valued = tf.get_static_value(upstream) is not None

        if static_valued:
            upstream = upstream.numpy()
            r_grad_sum = tf.constant(np.real(np.sum(upstream * np.conj(r_grad))))
            phi_grad_sum = tf.constant(np.real(np.sum(upstream * np.conj(phi_grad))))
        else:
            # NOTE: Possibly Tensorflow bug, cast needed.
            # cannot compute AddN as input #1(zero-based) was expected to be\
            #  a double tensor but is a float tensor [Op:AddN].
            # The bug does not occur with Displacement gradient for unknown reasons.
            r_grad_sum = tf.cast(
                tf.math.real(tf.reduce_sum(upstream * tf.math.conj(r_grad))), tf.float32
            )
            phi_grad_sum = tf.math.real(
                tf.reduce_sum(upstream * tf.math.conj(phi_grad))
            )

        return (r_grad_sum, phi_grad_sum)

    return squeezing_matrix_gradient
