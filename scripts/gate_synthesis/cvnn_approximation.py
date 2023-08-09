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

import gates_and_gradients

profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)


class CVNNApproximator:
    def __init__(
        self,
        cutoff=12,
        gate_cutoff=4,
        dtype=np.complex128,
        number_of_layers=15,
        number_of_steps=250,
        tolerance=0.05,
        optimizer_learning_rate=0.05,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        passive_sd=0.1,
        active_sd=0.001,
        isProfiling=False,
        seed=None,
    ):

        self._cutoff = cutoff
        self._gate_cutoff = gate_cutoff
        self._dtype = dtype
        self._number_of_layers = number_of_layers
        self._number_of_steps = number_of_steps
        self._tolerance = tolerance
        # TODO: Not working properly between unitaries.
        # self._optimizer = optimizer
        self._learning_rate = optimizer_learning_rate
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_learning_rate
        )
        self._passive_sd = passive_sd
        self._active_sd = active_sd
        self._seed = seed
        self._weights = None
        self._init_weights(seed)
        self._isProfiling = isProfiling

    def benchmark_gradient(self, target_kets):
        if self._isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)

        self._init_weights()
        start_time = time.time()

        with tf.GradientTape() as tape:
            cost, _ = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_1
            )

        print("Calculation took {} seconds".format(time.time() - start_time))

        start_time = time.time()
        gradients = tape.gradient(cost, self._weights)
        self._optimizer.apply_gradients(zip([gradients], [self._weights]))

        print("Gradient took {} seconds".format(time.time() - start_time))

        if self._isProfiling:
            tf.profiler.experimental.stop()

    def benchmark_two_gradients(self, target_kets):
        if self._isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)

        seed = 42

        self._init_weights(seed)
        start_time = time.time()

        with tf.GradientTape() as tape1:
            cost1, _ = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_default
            )
        print("Calculation 1 took {} seconds".format(time.time() - start_time))

        start_time = time.time()
        gradients1 = tape1.gradient(cost1, self._weights)
        self._optimizer.apply_gradients(zip([gradients1], [self._weights]))

        print("Gradient 1 took {} seconds".format(time.time() - start_time))

        self._init_weights(seed)
        start_time = time.time()

        with tf.GradientTape() as tape2:
            cost2, _ = self._cost_function(
                target_kets, self._calculate_unitary_with_cvnn_1
            )
        print("Calculation 2 took {} seconds".format(time.time() - start_time))

        start_time = time.time()
        gradients2 = tape2.gradient(cost2, self._weights)
        self._optimizer.apply_gradients(zip([gradients2], [self._weights]))
        print("Gradient 2 took {} seconds".format(time.time() - start_time))

        if self._isProfiling:
            tf.profiler.experimental.stop()

    def approximate_kets(self, kets):
        costs = []
        weights_list = []

        for i in range(len(kets)):
            self._init_weights()

            print("Starting unitary approximation No.{}".format(i + 1))
            start_time = time.time()

            cost, weights = self._perform_machine_learning(kets[i])

            print("Unitary approximation ended in:", time.time() - start_time)

            costs.append(cost)
            weights_list.append(weights)

        return costs, weights_list

    def _perform_machine_learning(self, target_kets):
        cost = min_cost = sys.maxsize - 1
        best_cvnn_weights = None

        if self._isProfiling:
            tf.profiler.experimental.start("logdir", options=profiler_options)

        i = 0
        while i < self._number_of_steps and cost > self._tolerance:
            with tf.GradientTape() as tape:
                cost, _ = self._cost_function(
                    target_kets, self._calculate_unitary_with_cvnn_default
                )

            if min_cost > cost.numpy():
                min_cost = cost.numpy()
                best_cvnn_weights = self._weights

            gradients = tape.gradient(cost, self._weights)
            self._optimizer.apply_gradients(zip([gradients], [self._weights]))

            print("Rep: {} Cost: {:.4f}".format(i + 1, cost))
            i += 1

        if self._isProfiling:
            tf.profiler.experimental.stop()

        best_cvnn_weights = np.asarray(best_cvnn_weights)
        return min_cost, best_cvnn_weights

    def _cost_function(self, target_kets, cvnn_calculator):
        unitary = cvnn_calculator()
        ket = unitary[:, : self._gate_cutoff].T
        overlaps = tf.math.real(tf.einsum("bi,bi->b", tf.math.conj(target_kets), ket))
        cost = tf.abs(tf.reduce_sum(overlaps - 1))

        return cost, ket

    def _calculate_unitary_with_cvnn_default(self):
        result_matrix = np.identity(self._cutoff, dtype=self._dtype)

        gate_creator = gates_and_gradients.AutoGradGateCreator()
        piquasso_gate_creator = gates_and_gradients.PiquassoGateCreator()

        for j in range(self._number_of_layers):

            phase_shifter_1_matrix = gate_creator.get_single_mode_phase_shift_matrix(
                self._weights[j, 0]
            )

            squeezing_matrix = (
                piquasso_gate_creator._fock_space.get_single_mode_squeezing_operator(
                    r=self._weights[j, 1], phi=self._weights[j, 2]
                )
            )

            phase_shifter_2_matrix = gate_creator.get_single_mode_phase_shift_matrix(
                self._weights[j, 3]
            )

            displacement_matrix = (
                piquasso_gate_creator._fock_space.get_single_mode_displacement_operator(
                    r=self._weights[j, 4], phi=self._weights[j, 5]
                )
            )

            kerr_matrix = gate_creator.get_single_mode_kerr_matrix(self._weights[j, 6])

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

    def _calculate_unitary_with_cvnn_1(self):
        result_matrix = np.identity(self._cutoff, dtype=self._dtype)

        piquasso_gate_creator = gates_and_gradients.PiquassoGateCreator()
        tf_gate_creator = gates_and_gradients.PureTensorFlowGateCreator()
        for j in range(self._number_of_layers):

            phase_shifter_1_matrix = tf_gate_creator.get_single_mode_phase_shift_matrix(
                self._weights[j, 0]
            )

            squeezing_matrix = (
                piquasso_gate_creator._fock_space.get_single_mode_squeezing_operator(
                    r=self._weights[j, 1], phi=self._weights[j, 2]
                )
            )

            phase_shifter_2_matrix = tf_gate_creator.get_single_mode_phase_shift_matrix(
                self._weights[j, 3]
            )

            displacement_matrix = (
                piquasso_gate_creator._fock_space.get_single_mode_displacement_operator(
                    r=self._weights[j, 4], phi=self._weights[j, 5]
                )
            )

            kerr_matrix = tf_gate_creator.get_single_mode_kerr_matrix(
                self._weights[j, 6]
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

    def _init_weights(self, seed=None):
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            tf.random.set_seed(seed)

        # squeezing
        sq_r = tf.random.normal(shape=[self._number_of_layers], stddev=self._active_sd)
        sq_phi = tf.random.normal(
            shape=[self._number_of_layers], stddev=self._passive_sd
        )

        # displacement gate
        d_r = tf.random.normal(shape=[self._number_of_layers], stddev=self._active_sd)
        d_phi = tf.random.normal(
            shape=[self._number_of_layers], stddev=self._passive_sd
        )

        # rotation gates
        r1 = tf.random.normal(shape=[self._number_of_layers], stddev=self._passive_sd)
        r2 = tf.random.normal(shape=[self._number_of_layers], stddev=self._passive_sd)

        # kerr gate
        kappa = tf.random.normal(shape=[self._number_of_layers], stddev=self._active_sd)

        weights = tf.convert_to_tensor(
            [r1, sq_r, sq_phi, r2, d_r, d_phi, kappa], dtype=np.float64
        )

        self._weights = tf.Variable(tf.transpose(weights))
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
