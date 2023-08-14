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
import tensorflow as tf
from numba import jit
import time

import piquasso as pq


def maybe_convert_to_numpy(value):
    """
    Used when we need to index arrays directly and assign values to them.
    """
    return value.numpy() if tf.is_tensor(value) else value


class PiquassoGateCreator:
    def __init__(self, cutoff=12, dtype=np.complex128):
        self._cutoff = cutoff
        self._dtype = dtype

        calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
        config = pq.Config(dtype=np.float64)
        self._fock_space = pq._math.fock.FockSpace(
            d=1, cutoff=cutoff, calculator=calculator, config=config
        )

    # TODO Kerr and Phaseshifter depending on what to use here.


class AutoGradGateCreator:
    def __init__(self, cutoff=12, dtype=np.complex128):
        self._cutoff = cutoff
        self._dtype = dtype

    def get_single_mode_squeezing_operator(self, r, phi):
        # The gradient is 0 because tensorflow couldn't be used here due to the necessity of assignments
        raise NotImplementedError

    def get_single_mode_displacement_operator(self, r, phi):
        # The gradient is 0 because tensorflow couldn't be used here due to the necessity of assignments
        raise NotImplementedError

    def get_single_mode_kerr_matrix(self, kappa: float):
        coefficients = tf.exp(
            1j * kappa * tf.math.pow(tf.cast(tf.range(self._cutoff), tf.complex128), 2)
        )
        return tf.linalg.diag(coefficients)

    def get_single_mode_phase_shift_matrix(self, phi: float):
        coefficients = tf.math.exp(
            1j * phi * tf.cast(tf.range(self._cutoff), tf.complex128)
        )
        return tf.linalg.diag(coefficients)


class PureTensorFlowGateCreator:
    def __init__(self, cutoff=12, dtype=tf.complex128):
        self._cutoff = cutoff
        self._dtype = dtype

    @tf.custom_gradient
    def get_single_mode_squeezing_operator(self, r, phi):
        sechr = 1.0 / tf.math.cosh(r)
        A = tf.math.exp(1j * phi) * tf.math.tanh(r)
        a_conj = tf.math.conj(A)
        transformation = [[0 for _ in range(self._cutoff)] for _ in range(self._cutoff)]
        transformation[0][0] = tf.sqrt(sechr)

        fock_indices = tf.math.sqrt(tf.range(self._cutoff, dtype=tf.float64))

        for index in range(2, self._cutoff, 2):
            transformation[index][0] = (
                -fock_indices[index - 1]
                / fock_indices[index]
                * (transformation[index - 2][0] * A)
            )

        for row in range(0, self._cutoff):
            for col in range(1, self._cutoff):
                if (row + col) % 2 == 0:
                    transformation[row][col] = (
                        1
                        / fock_indices[col]
                        * (
                            (fock_indices[row] * transformation[row - 1][col - 1] * sechr)
                            + (
                                fock_indices[col - 1]
                                * a_conj
                                * transformation[row][col - 2]
                            )
                        )
                    )

        def get_squeezing_matrix_gradient(upstream):
            breakpoint()
            start_time = time.time()
            sechr = 1 / tf.math.cosh(r)
            tanhr = tf.math.tanh(r)

            index_sqrts = tf.math.sqrt(tf.range(self._cutoff, dtype=self._dtype))

            falling_index_sqrts = index_sqrts * tf.roll(index_sqrts, 1)

            row_sqrts = falling_index_sqrts * tf.math.exp(1j * phi)
            col_sqrts = falling_index_sqrts * tf.math.exp(-1j * phi)

            row_rolled_transformation = tf.roll(transformation, 2, axis=0)
            col_rolled_transformation = tf.roll(transformation, 2, axis=1)

            row_term = (row_sqrts * row_rolled_transformation.T).T
            col_term = col_sqrts * col_rolled_transformation

            # NOTE: This algorithm rolls the last and penultimate elements of the
            # transormation matrix to the 1st and 0th place, but the 0th and 1st element of
            # `row_sqrts` and `col_sqrts` is always zero, so it is fine.
            phi_grad = -0.5j * tanhr * (row_term + col_term)

            diagonally_rolled_transformation = tf.roll(transformation, (1, 1), axis=(0, 1))

            r_grad = (
                (-tanhr * 0.5) * transformation
                - (sechr * tanhr)
                * tf.tensordot(index_sqrts, index_sqrts, axes=0)
                * diagonally_rolled_transformation
                - (sechr**2 * 0.5) * (row_term - col_term)
            )

            # NOTE: Possibly Tensorflow bug, cast needed.
            # cannot compute AddN as input #1(zero-based) was expected to be\
            #  a double tensor but is a float tensor [Op:AddN].
            # The bug does not occur with Displacement gradient for unknown reasons.
            r_grad_sum = tf.math.real(tf.reduce_sum(upstream * tf.math.conj(r_grad)))
            phi_grad_sum = tf.math.real(
                tf.reduce_sum(upstream * tf.math.conj(phi_grad))
            )

            return (r_grad_sum, phi_grad_sum)

        return transformation, get_squeezing_matrix_gradient

    @tf.custom_gradient
    def get_single_mode_displacement_operator(self, r, phi):
        raise NotImplementedError

        def get_displacement_matrix_gradient(upstream):
            raise NotImplementedError

    @tf.custom_gradient
    def get_single_mode_phase_shift_matrix(self, phi):
        cutoff_range = tf.range(self._cutoff)
        coefficients = tf.exp(1j * phi * cutoff_range)
        transformation = tf.linalg.diag(coefficients)

        def get_single_mode_phase_shift_matrix_grad(upstream):
            phi_grad = tf.exp(1j * phi * cutoff_range) * cutoff_range * 1j
            phi_grad = tf.linalg.diag(phi_grad)
            return tf.math.real(tf.reduce_sum(upstream * tf.math.conj(phi_grad)))

        return transformation, get_single_mode_phase_shift_matrix_grad

    @tf.custom_gradient
    def get_single_mode_kerr_matrix(self, kappa):
        # cast due to the tf.math.pow
        cutoff_range = tf.cast(tf.range(self._cutoff), tf.complex128)
        coefficients = tf.exp(1j * kappa * tf.math.pow(cutoff_range, 2))
        transformation = tf.linalg.diag(coefficients)

        def get_single_mode_kerr_matrix_grad(upstream):
            coefficients = (
                tf.exp(1j * kappa * tf.math.pow(cutoff_range, 2))
                * 1j
                * tf.math.pow(cutoff_range, 2)
            )
            kappa_grad = tf.linalg.diag(coefficients)

            return tf.math.real(tf.reduce_sum(upstream * tf.math.conj(kappa_grad)))

        return transformation, get_single_mode_kerr_matrix_grad

@tf.function(jit_compile=True)
def get_single_mode_squeezing_operator_tf_func_decorator(r, phi, cutoff):
    phi = tf.cast(phi, tf.complex128)
    r = tf.cast(r, tf.complex128)
    sechr = 1.0 / tf.math.cosh(r)
    A = tf.math.exp(1j * phi) * tf.math.tanh(r)
    a_conj = tf.math.conj(A)
    transformation = [[0j for _ in range(cutoff)] for _ in range(cutoff)]
    transformation[0][0] = tf.math.sqrt(sechr)

    fock_indices = tf.math.sqrt(tf.cast(tf.range(cutoff, dtype=tf.float64), tf.complex128))

    for index in range(2, cutoff, 2):
        transformation[index][0] = (
            -fock_indices[index - 1]
            / fock_indices[index]
            * (transformation[index - 2][0] * A)
        )

    for row in range(0, cutoff):
        for col in range(1, cutoff):
            if (row + col) % 2 == 0:
                transformation[row][col] = (
                    1
                    / fock_indices[col]
                    * (
                        (fock_indices[row] * transformation[row - 1][col - 1] * sechr)
                        + (
                            fock_indices[col - 1]
                            * a_conj
                            * transformation[row][col - 2]
                        )
                    )
                )

    return transformation

@tf.function(jit_compile=True)
def get_single_mode_displacement_operator_tf_func_decorator(r, phi, cutoff):
    phi = tf.cast(phi, tf.complex128)
    r = tf.cast(r, tf.complex128)
    fock_indices = tf.math.sqrt(tf.cast(tf.range(cutoff, dtype=tf.float64), tf.complex128))
    displacement = r * tf.math.exp(1j * phi)
    displacement_conj = tf.math.conj(displacement)
    transformation = [[0j for _ in range(cutoff)] for _ in range(cutoff)]
    transformation[0][0] = tf.math.exp(-0.5 * r**2)
    for row in range(1, cutoff):
        transformation[row][0] = (
            displacement / fock_indices[row] * transformation[row - 1][0]
        )
    for row in range(cutoff):
        for col in range(1, cutoff):
            transformation[row][col] = (
                -displacement_conj
                / fock_indices[col]
                * transformation[row][col - 1]
            ) + (
                fock_indices[row] / fock_indices[col] * transformation[row - 1][col - 1]
            )

    return transformation

@tf.function(jit_compile=True)
def get_single_mode_kerr_matrix_tf_func_decorator(kappa, cutoff):
    coefficients = tf.math.exp(
        1j * tf.cast(kappa, tf.complex128) * tf.math.pow(tf.cast(tf.range(cutoff), tf.complex128), 2)
    )
    return tf.linalg.diag(coefficients)


@tf.function(jit_compile=True)
def get_single_mode_phase_shift_matrix_tf_func_decorator(phi, cutoff):
    coefficients = tf.math.exp(
        1j * tf.cast(phi, tf.complex128) * tf.cast(tf.range(cutoff), tf.complex128)
    )
    return tf.linalg.diag(coefficients)


class TensorFlowFunctionGateCreator:
    def __init__(self, cutoff=12, dtype=tf.complex128):
        self._cutoff = cutoff
        self._dtype = dtype


    def get_single_mode_squeezing_operator(self, r, phi):
        return get_single_mode_squeezing_operator_tf_func_decorator(r, phi, self._cutoff)

    def get_single_mode_displacement_operator(self, r, phi):
        return get_single_mode_displacement_operator_tf_func_decorator(r, phi, self._cutoff)

    def get_single_mode_phase_shift_matrix(self, phi):
        return get_single_mode_phase_shift_matrix_tf_func_decorator(phi, self._cutoff)

    def get_single_mode_kerr_matrix(self, kappa):
        return get_single_mode_kerr_matrix_tf_func_decorator(kappa, self._cutoff)