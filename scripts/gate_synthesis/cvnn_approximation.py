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

import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

import gates_and_gradients

from param_config import *

number_of_cvnn_layers
tnp.experimental_enable_numpy_behavior()
profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)

global cvnn_weights
cvnn_weights = None
def init_weights(seed=None):
    if seed is not None:
        seed = seed
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # squeezing
    sq_r = np.random.normal(size=[number_of_cvnn_layers], scale=active_sd)
    sq_phi = np.random.normal(
        size=[number_of_cvnn_layers], scale=passive_sd
    )
    # displacement gate
    d_r = np.random.normal(size=[number_of_cvnn_layers], scale=active_sd)
    d_phi = np.random.normal(
        size=[number_of_cvnn_layers], scale=passive_sd
    )
    # rotation gates
    r1 = np.random.normal(size=[number_of_cvnn_layers], scale=passive_sd)
    r2 = np.random.normal(size=[number_of_cvnn_layers], scale=passive_sd)
    # kerr gate
    kappa = np.random.normal(size=[number_of_cvnn_layers], scale=active_sd)

    weights = tf.convert_to_tensor(
        [r1, sq_r, sq_phi, r2, d_r, d_phi, kappa], dtype=np.float64
    )
    cvnn_weights = tf.Variable(tf.transpose(weights))
    cvnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cvnn_learning_rate)

init_weights(seed)

class CVNNApproximator:
    def __init__(
        self,
        cutoff=12,
        gate_cutoff=4,
        dtype=np.complex128,
        number_of_steps=250,
        tolerance=0.05,
        optimizer_learning_rate=0.05,
        isProfiling=False,
        seed=None,
    ):

        self._cutoff = cutoff
        self._gate_cutoff = gate_cutoff
        self._dtype = dtype
        self._number_of_steps = number_of_steps
        self._tolerance = tolerance
        self._learning_rate = optimizer_learning_rate
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_learning_rate
        )
        self._seed = seed
        self._isProfiling = isProfiling

    def benchmark_gradient(self, target_kets):
        if self._isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)

        init_weights()
        start_time = time.time()

        with tf.GradientTape() as tape:
            cost, _ = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_default
            )
        calc_time = time.time() - start_time
        print("Calculation took {} seconds".format(calc_time))

        start_time = time.time()
        gradients = tape.gradient(cost, cvnn_weights)
        self._optimizer.apply_gradients(zip([gradients], [cvnn_weights]))
        custom_apply_gradient(gradients, cvnn_weights, self._learning_rate)

        grad_time = time.time() - start_time
        print("Gradient took {} seconds".format(grad_time))

        if self._isProfiling:
            tf.profiler.experimental.stop()

        return calc_time, grad_time

    def benchmark_two_gradients(self, target_kets):

        seed = 42
        if self._isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)
        init_weights(seed)
        start_time = time.time()

        with tf.GradientTape() as tape1:
            cost1, ket_val1 = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_vectorized
            )

        calc1_time = time.time() - start_time
        print("Calculation 1 took {} seconds".format(calc1_time))

        start_time = time.time()
        gradients1 = tape1.gradient(cost1, cvnn_weights)
        self._optimizer.apply_gradients(zip([gradients1], [cvnn_weights]))
        grad1_time = time.time() - start_time
        print("Gradient 1 took {} seconds".format(grad1_time))

        self._init_weights(seed)
        start_time = time.time()

        with tf.GradientTape() as tape2:
            cost2, ket_val2 = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_1
            )
        calc2_time = time.time() - start_time
        print("Calculation 2 took {} seconds".format(calc2_time))

        start_time = time.time()
        gradients2 = tape2.gradient(cost2, cvnn_weights)
        self._optimizer.apply_gradients(zip([gradients2], [cvnn_weights]))
        grad2_time = time.time() - start_time
        print("Gradient 2 took {} seconds".format(grad2_time))
        assert np.allclose(gradients1, gradients2)
        assert np.allclose(ket_val1, ket_val2)
        if self._isProfiling:
            tf.profiler.experimental.stop()

        return (calc1_time, grad1_time, calc2_time, grad2_time)


def calculate_unitary_with_cvnn_default(weights=None):
    if weights is not None:
        cvnn_weights = weights

    result_matrix = np.identity(cutoff, dtype=dtype)

    gate_creator = gates_and_gradients.AutoGradGateCreator()
    piquasso_gate_creator = gates_and_gradients.PiquassoGateCreator()

    for j in range(number_of_cvnn_layers):

        phase_shifter_1_matrix = gate_creator.get_single_mode_phase_shift_matrix(
            cvnn_weights[j, 0]
        )

        squeezing_matrix = (
            piquasso_gate_creator._fock_space.get_single_mode_squeezing_operator(
                r=cvnn_weights[j, 1], phi=cvnn_weights[j, 2]
            )
        )

        phase_shifter_2_matrix = gate_creator.get_single_mode_phase_shift_matrix(
            cvnn_weights[j, 3]
        )

        displacement_matrix = (
            piquasso_gate_creator._fock_space.get_single_mode_displacement_operator(
                r=cvnn_weights[j, 4], phi=cvnn_weights[j, 5]
            )
        )

        kerr_matrix = gate_creator.get_single_mode_kerr_matrix(cvnn_weights[j, 6])
        result_matrix = result_matrix = tf.matmul(
            tf.matmul(
                tf.matmul(
                    tf.matmul(
                        tf.matmul(kerr_matrix, displacement_matrix),
                        phase_shifter_2_matrix,
                    ),
                    squeezing_matrix,
                ),
                phase_shifter_1_matrix,
            ),
            result_matrix,
        )

    return result_matrix


def calculate_unitary_with_cvnn_for_solution2(weights=None):
    if weights is not None:
            cvnn_weights = weights

    result_matrix = tf.linalg.eye(cutoff, dtype=dtype)

    for j in range(number_of_cvnn_layers):
        phase_shifter_1_matrix = gates_and_gradients.get_single_mode_phase_shift_matrix_tf_func_decorator(
            cvnn_weights[j, 0]
        )
        squeezing_matrix = gates_and_gradients.get_single_mode_squeezing_operator_tf_func_decorator(
                (cvnn_weights[j, 1], cvnn_weights[j, 2])
            )
        phase_shifter_2_matrix = gates_and_gradients.get_single_mode_phase_shift_matrix_tf_func_decorator(
            cvnn_weights[j, 3]
        )
        displacement_matrix = (
            gates_and_gradients.get_single_mode_displacement_operator_tf_func_decorator(
                (cvnn_weights[j, 4], cvnn_weights[j, 5])
            )
        )
        kerr_matrix = gates_and_gradients.get_single_mode_kerr_matrix_tf_func_decorator(
            cvnn_weights[j, 6]
        )
        result_matrix = result_matrix = tf.matmul(
            tf.matmul(
                tf.matmul(
                    tf.matmul(
                        tf.matmul(kerr_matrix, displacement_matrix),
                        phase_shifter_2_matrix,
                    ),
                    squeezing_matrix,
                ),
                phase_shifter_1_matrix,
            ),
            result_matrix,
        )
    return result_matrix

@tf.function(jit_compile=True)
def calculate_unitary_with_cvnn_vectorized(weights=None):
        if weights is not None:
             cvnn_weights = weights

        start_time = time.time()
        phase_shifter1_vectorization = tf.vectorized_map(
            gates_and_gradients.get_single_mode_phase_shift_matrix_tf_func_decorator,
            cvnn_weights[:, 0]
        )
        print("Phaseshifter took {} seconds".format(time.time() - start_time))
        start_time = time.time()
        squeeze_vectorization = tf.vectorized_map(
            gates_and_gradients.get_single_mode_squeezing_operator_tf_func_decorator,
            (cvnn_weights[:, 1], cvnn_weights[:, 2])
        )
        print("Squeezing took {} seconds".format(time.time() - start_time))
        phase_shifter2_vectorization = tf.vectorized_map(
            gates_and_gradients.get_single_mode_phase_shift_matrix_tf_func_decorator,
            cvnn_weights[:, 3]
        )
        start_time = time.time()
        displacement_vectorization = tf.vectorized_map(
            gates_and_gradients.get_single_mode_displacement_operator_tf_func_decorator,
            (cvnn_weights[:, 4], cvnn_weights[:, 5])
        )
        print("Displacement took {} seconds".format(time.time() - start_time))
        start_time = time.time()
        kerr_vectorization = tf.vectorized_map(
            gates_and_gradients.get_single_mode_kerr_matrix_tf_func_decorator,
            cvnn_weights[:, 6]
        )
        print("Kerr took {} seconds".format(time.time() - start_time))

        start_time = time.time()
        vectorized_result_matrix = tf.reverse(
            tf.einsum(
            "abc,acd,ade,aef,afg->abg",
            kerr_vectorization,
            displacement_vectorization,
            phase_shifter2_vectorization,
            squeeze_vectorization,
            phase_shifter1_vectorization
        ), [0])
        print("Einsum took {} seconds".format(time.time() - start_time))
        #new_vectorized_result_matrix = tf.einsum("iab,ibc->ac", vectorized_result_matrix, vectorized_result_matrix)
        #new_vectorized_result_matrix = np.linalg.multi_dot(vectorized_result_matrix)
        start_time = time.time()
        result_matrix = tf.linalg.eye(cutoff, dtype=dtype)
        for i in range(number_of_cvnn_layers):
            result_matrix = tf.matmul(result_matrix, vectorized_result_matrix[i])
        print("Last for took {} seconds".format(time.time() - start_time))
        return result_matrix


def approximate_kets(kets):
        costs = []
        weights_list = []

        for i in range(len(kets)):
            init_weights()

            print("Starting unitary approximation No.{}".format(i + 1))
            start_time = time.time()

            cost, weights = perform_machine_learning(kets[i])

            print("Unitary approximation ended in:", time.time() - start_time)

            costs.append(cost)
            weights_list.append(weights)

        return costs, weights_list


def perform_machine_learning(target_kets):
        cost = min_cost = sys.maxsize - 1
        best_cvnn_weights = None

        if isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)

        i = 0
        while i < number_of_cvnn_steps and cost > cvnn_tolerance:
            with tf.GradientTape() as tape:
                cost, _ = cost_function(
                    target_kets, calculate_unitary_with_cvnn_default
                )

            if min_cost > cost.numpy():
                min_cost = cost.numpy()
                best_cvnn_weights = cvnn_weights

            gradients = tape.gradient(cost, cvnn_weights)
            cvnn_optimizer.apply_gradients(zip([gradients], [cvnn_weights]))

            print("Rep: {} Cost: {:.4f}".format(i + 1, cost))
            i += 1

        if isProfiling:
            tf.profiler.experimental.stop()

        best_cvnn_weights = np.asarray(best_cvnn_weights)
        return min_cost, best_cvnn_weights


def cost_function(target_kets, cvnn_calculator):
        unitary = cvnn_calculator()
        ket = tf.transpose(unitary[:, :gate_cutoff])
        overlaps = tf.math.real(tf.einsum("bi,bi->b", tf.math.conj(target_kets), ket))
        cost = tf.abs(tf.reduce_sum(overlaps - 1))

        return cost, ket

@tf.function(jit_compile=True)
def custom_apply_gradient(gradients, variables, learning_rate):
    return variables - gradients * learning_rate;
