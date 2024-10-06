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

import numpy as np

from typing import Callable

from piquasso.api.connector import BaseConnector


def create_single_mode_displacement_gradient(
    r: float,
    phi: float,
    cutoff: int,
    transformation: np.ndarray,
    connector: BaseConnector,
) -> Callable:
    def displacement_matrix_gradient(upstream):
        np = connector.fallback_np

        index_sqrts = np.sqrt(range(cutoff))

        row_sqrts = index_sqrts * np.exp(1j * phi)
        col_sqrts = index_sqrts * np.exp(-1j * phi)

        row_rolled_transformation = np.roll(transformation, 1, axis=0)
        col_rolled_transformation = np.roll(transformation, 1, axis=1)

        row_term = (row_sqrts * row_rolled_transformation.T).T
        col_term = col_sqrts * col_rolled_transformation

        # NOTE: This algorithm rolls the last elements of the transormation matrix to
        # the 0th place, but the 0th element of `row_sqrts` and `col_sqrts` is always
        # zero, so it is fine.
        phi_grad = (row_term + col_term) * r * 1j

        r_grad = -r * transformation + row_term - col_term

        tf = connector._tf
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
    connector: BaseConnector,
) -> Callable:
    def squeezing_matrix_gradient(upstream):
        np = connector.fallback_np
        tf = connector._tf

        sechr = 1 / np.cosh(r)
        tanhr = np.tanh(r)

        index_sqrts = np.sqrt(np.arange(cutoff, dtype=np.complex128))

        falling_index_sqrts = index_sqrts * np.roll(index_sqrts, 1)

        row_sqrts = falling_index_sqrts * np.exp(1j * phi)
        col_sqrts = falling_index_sqrts * np.exp(-1j * phi)

        row_rolled_transformation = np.roll(transformation, 2, axis=0)
        col_rolled_transformation = np.roll(transformation, 2, axis=1)

        row_term = (row_sqrts * row_rolled_transformation.T).T
        col_term = col_sqrts * col_rolled_transformation

        # NOTE: This algorithm rolls the last and penultimate elements of the
        # transormation matrix to the 1st and 0th place, but the 0th and 1st element of
        # `row_sqrts` and `col_sqrts` is always zero, so it is fine.
        phi_grad = -0.5j * tanhr * (row_term + col_term)

        diagonally_rolled_transformation = np.roll(transformation, (1, 1), axis=(0, 1))

        r_grad = (
            (-tanhr * 0.5) * transformation
            - (sechr * tanhr)
            * np.outer(index_sqrts, index_sqrts)
            * diagonally_rolled_transformation
            - (sechr**2 * 0.5) * (row_term - col_term)
        )

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

    return squeezing_matrix_gradient
