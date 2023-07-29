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

sys.path.append(".")
import time

import numpy as np
import tensorflow as tf

import piquasso as pq
import normal_ordering


profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)

# Cutoff dimension
cutoff = 10

# gate cutoff
gate_cutoff = 4

# Number of layers
layer_amount = 15

# Number of unitaries to generate data with
unitary_amount = 100

# Number of steps in optimization routine performing gradient descent
number_of_steps = 300

# Learning rate
lr = 0.05

# Standard deviation of initial parameters
passive_sd = 0.1
active_sd = 0.001

d = 1

# set the random seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# squeeze gate
sq_r = tf.random.normal(shape=[layer_amount], stddev=active_sd)
sq_phi = tf.random.normal(shape=[layer_amount], stddev=passive_sd)

# displacement gate
d_r = tf.random.normal(shape=[layer_amount], stddev=active_sd)
d_phi = tf.random.normal(shape=[layer_amount], stddev=passive_sd)

# rotation gates
r1 = tf.random.normal(shape=[layer_amount], stddev=passive_sd)
r2 = tf.random.normal(shape=[layer_amount], stddev=passive_sd)

# kerr gate
kappa = tf.random.normal(shape=[layer_amount], stddev=active_sd)


weights = tf.convert_to_tensor(
    [r1, sq_r, sq_phi, r2, d_r, d_phi, kappa], dtype=np.float64
)
weights = tf.Variable(tf.transpose(weights))

calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
tnp = calculator.np  # gradient had None values
config = pq.Config(dtype=np.complex128)
fock_space = pq._math.fock.FockSpace(
    d=d, cutoff=cutoff, calculator=calculator, config=config
)


def get_single_mode_kerr_matrix(xi: float):
    coefficients = tnp.exp(1j * xi * tnp.power(tnp.arange(cutoff), 2))
    return tnp.diag(coefficients)


def get_single_mode_phase_shift_matrix(phi: float):
    coefficients = tnp.exp(1j * phi * tnp.arange(cutoff))
    return tnp.diag(coefficients)


# define random unitaries up to gate_cutoff
order = 6
unitaries = []
unitary_coeffs = []
target_kets_list = []
for _ in range(unitary_amount):
    random_unitary, unitary_coeffs = normal_ordering.generate_unitary(
        order, gate_cutoff, seed
    )
    unitaries.append(random_unitary)
    unitary_coeffs.append(unitary_coeffs)
    # extend unitary up to cutoff
    target_unitary = np.identity(cutoff, dtype=np.complex128)
    target_unitary[:gate_cutoff, :gate_cutoff] = random_unitary

    target_kets = np.array([target_unitary[:, i] for i in np.arange(gate_cutoff)])
    target_kets = tf.constant(target_kets, dtype=tf.complex128)
    target_kets_list.append(target_kets)

# For json conversions
unitaries = np.asarray(unitaries)
unitary_coeffs = np.asarray(unitary_coeffs)
target_kets_list = np.asarray(target_kets_list)


def approx_unitary_with_cvnn(layer_params, number_of_layers):
    result_matrix = tnp.identity(cutoff, dtype=np.complex128)

    for j in range(number_of_layers):
        phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(layer_params[j, 0])
        squeezing_matrix = fock_space.get_single_mode_squeezing_operator(
            r=layer_params[j, 1], phi=layer_params[j, 2]
        )
        phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(layer_params[j, 3])
        displacement_matrix = fock_space.get_single_mode_displacement_operator(
            r=layer_params[j, 4], phi=layer_params[j, 5]
        )
        kerr_matrix = get_single_mode_kerr_matrix(layer_params[j, 6])
        result_matrix = (
            kerr_matrix
            @ displacement_matrix
            @ phase_shifter_2_matrix
            @ squeezing_matrix
            @ phase_shifter_1_matrix
            @ result_matrix
        )

    return result_matrix


def cost_fn(weights):

    # Run engine
    unitary = approx_unitary_with_cvnn(weights, layer_amount)

    ket = unitary[:, :gate_cutoff].T
    # overlaps
    overlaps = tf.math.real(tf.einsum("bi,bi->b", tf.math.conj(target_kets), ket))
    # Objective function to minimize
    cost = tf.abs(tf.reduce_sum(overlaps - 1))

    return cost, overlaps, ket


# Just to be able to index them within for loops
approximated_unitaries = [0 for _ in range(unitary_amount)]
min_costs = [sys.maxsize - 1 for _ in range(unitary_amount)]
best_cvnn_weights = [0 for _ in range(unitary_amount)]
# tf.profiler.experimental.start("logdir", options=profiler_options)
# start_time = time.time()
# Run data generation
for current_unitary in range(unitary_amount):
    # Run optimization
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    for i in range(number_of_steps):

        # one repetition of the optimization
        with tf.GradientTape() as tape:
            cost, overlaps_val, ket_val = cost_fn(weights)

        # Store best values to use as data
        if min_costs[current_unitary] > cost.numpy():
            min_costs[current_unitary] = cost.numpy()
            best_cvnn_weights[current_unitary] = weights
            approximated_unitaries[current_unitary] = ket_val

        # calculate the mean overlap
        # This gives us an idea of how the optimization is progressing
        mean_overlap_val = np.mean(overlaps_val)

        # one repetition of the optimization
        gradients = tape.gradient(cost, weights)
        opt.apply_gradients(zip([gradients], [weights]))

        # Prints progress at every rep
        if i % 1 == 0:
            # print progress
            print(
                "Rep: {} Cost: {:.4f} Mean overlap: {:.4f}".format(
                    i, cost, mean_overlap_val
                )
            )

# print("COMPLETE RUNTIME: ", time.time() - start_time)
# tf.profiler.experimental.stop()
approximated_unitaries = np.asarray(approximated_unitaries)
best_cvnn_weights = np.asarray(best_cvnn_weights)
import json

# NOTE: https://stackoverflow.com/questions/57826092/how-to-convert-string-formed-by-numpy-array2string-back-to-array
# Eval is probably the simplest solution
data = {
    "general_info": {
        "cutoff": cutoff,
        "mode_amount": d,
        "gate_cutoff": gate_cutoff,
        "number_of_layers": layer_amount,
        "number_of_steps": number_of_steps,
        "number_of_unitaries": unitary_amount,
        "learnin_rate": lr,
        "active_sd": active_sd,
        "passive_sd": passive_sd,
        "seed": seed,
        "hamiltonian_order": order,
        "optimizer": opt.__str__(),
    },
    "data": {
        "target_coeffs": np.array2string(unitary_coeffs, separator=","),
        "target_unitaries": np.array2string(unitaries, separator=","),
        "target_kets": np.array2string(target_kets_list, separator=","),
        "approx_unitaries": np.array2string(approximated_unitaries, separator=","),
        "cvnn_weights": np.array2string(best_cvnn_weights, separator=","),
        "min_cost": min_costs,
    },
}

import datetime

json_file = open(
    "scripts/gate_synthesis/cvnn_approximations/train_data{}.json".format(
        datetime.datetime.today()
    ),
    "w",
)
json.dump(data, json_file, indent=4)
json_file.close()
