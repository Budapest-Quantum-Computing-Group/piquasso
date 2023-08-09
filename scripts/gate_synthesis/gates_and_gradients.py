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
    def __init__(self, cutoff=12, dtype=np.complex128):
        self._cutoff = cutoff
        self._dtype = dtype

    @tf.custom_gradient
    def get_single_mode_squeezing_operator(self, r, phi):
        raise NotImplementedError

        def get_squeezing_matrix_gradient(upstream):
            raise NotImplementedError

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
