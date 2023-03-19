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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import piquasso as pq
from piquasso._math.fock import cutoff_cardinality
import tensorflow as tf
import time
import numpy as np


def _get_number_of_beamsplitters(d: int):
    number_of_beamsplitters: int
    if d % 2 == 0:
        number_of_beamsplitters = (d // 2) ** 2
        number_of_beamsplitters += ((d - 1) // 2) * (d // 2)
    else:
        number_of_beamsplitters = ((d - 1) // 2) * d

    return number_of_beamsplitters


def create_layer_parameters(d: int, number_of_layers: int,):
    number_of_beamsplitters = _get_number_of_beamsplitters(d)

    dtype = tf.float64

    thetas_1 = tf.random.uniform(shape=[number_of_layers, number_of_beamsplitters], dtype=dtype, maxval=np.pi*2)
    phis_1 = tf.random.uniform(shape=[number_of_layers, d-1], dtype=dtype, maxval=np.pi*2)

    squeezings = tf.ones(shape=[number_of_layers, d], dtype=dtype)*0.1

    thetas_2 = tf.random.uniform(shape=[number_of_layers, number_of_beamsplitters], dtype=dtype, maxval=np.pi*2)
    phis_2 = tf.random.uniform(shape=[number_of_layers, d-1], dtype=dtype, maxval=np.pi*2)

    displacements_r = tf.ones(shape=[number_of_layers, d], dtype=dtype)*0.1
    displacements_phi = tf.zeros(shape=[number_of_layers, d], dtype=dtype)

    kappas = tf.random.uniform(shape=[number_of_layers, d], dtype=dtype)

    weights = tf.concat(
        [thetas_1, phis_1, squeezings, thetas_2, phis_2, displacements_r, displacements_phi, kappas], axis=1
    )

    return tf.Variable(weights)


d = 9
cutoff = 5
min_cutoff = 2
max_cutoff = 10
number_of_layers = 12

for number_of_layers in range(1, 6):
    print("###############")
    print("CUTOFF:", cutoff)
    target_state_vector = np.zeros(cutoff_cardinality(cutoff=cutoff, d=d), dtype=complex)
    target_state_vector[1] = 1.0
    target_state = tf.constant(target_state_vector, dtype=tf.complex128)

    simulator = pq.TensorflowPureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

    parameters = create_layer_parameters(d, number_of_layers)


    with tf.GradientTape() as tape:
        def create_interferometer(thetas, phis):
            with pq.Program() as interferometer:
                i = 0
                for col in range(d):
                    if col % 2 == 0:
                        for mode in range(0, d - 1, 2):
                            modes = (mode, mode + 1)
                            pq.Q(*modes) | pq.Beamsplitter(thetas[i], phi=0.0)
                            i += 1

                    if col % 2 == 1:
                        for mode in range(1, d - 1, 2):
                            modes = (mode, mode + 1)
                            pq.Q(*modes) | pq.Beamsplitter(thetas[i], phi=0.0)
                            i += 1

                for i in range(d - 1):
                    pq.Q(i) | pq.Phaseshifter(phis[i])

            return interferometer

        def create_layer(single_layer_parameters):
            k = _get_number_of_beamsplitters(d)

            thetas_1 = single_layer_parameters[:k]
            phis_1 = single_layer_parameters[k:k+d-1]

            squeezings = single_layer_parameters[k+d-1: k+2*d-1]

            thetas_2 = single_layer_parameters[k+2*d-1: 2*k+2*d-1]
            phis_2 = single_layer_parameters[2*k+2*d-1: 2*k+3*d-2]

            displacements_r = single_layer_parameters[2*k+3*d-2: 2*k+4*d-2]
            displacements_phi = single_layer_parameters[2*k+4*d-2: 2*k+5*d-2]

            kappas = single_layer_parameters[2*k+5*d-2: 2*k+6*d-2]

            first_interferometer = create_interferometer(thetas_1, phis_1)
            second_interferometer = create_interferometer(thetas_2, phis_2)

            with pq.Program() as single_layer:
                pq.Q(all) | first_interferometer

                pq.Q(all) | pq.Squeezing(squeezings)

                pq.Q(all) | second_interferometer

                for i in range(d - 1):
                    pq.Q(i) | pq.Phaseshifter(phis_2[i])

                pq.Q(all) | pq.Displacement(r=displacements_r, phi=displacements_phi)
                pq.Q(all) | pq.Kerr(kappas)

            return single_layer

        layers = [create_layer(parameters[i]) for i in range(parameters.shape[0])]

        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            for layer in layers:
                pq.Q(all) | layer


        state = simulator.execute(program).state

        state_vector = state._state_vector

        cost = tf.reduce_sum(tf.abs(target_state_vector - state_vector))


    start_time = time.time()
    gradient = tape.gradient(cost, parameters)

    #print(gradient)

    print("GRADIENT CALCULATION TIME:", time.time() - start_time)

    file = open("lofasz.txt", "r")
    lines = file.readlines()
    result = 1
    for line in lines:
        result *= float(line)

    # print(result)
    file.close()
    os.remove("lofasz.txt")
