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

import piquasso as pq
from piquasso._math.fock import cutoff_cardinality
import tensorflow as tf
import time
import numpy as np


def create_layer_parameters(d: int):
    number_of_beamsplitters: int
    if d % 2 == 0:
        number_of_beamsplitters = (d // 2) ** 2
        number_of_beamsplitters += ((d - 1) // 2) * (d // 2)
    else:
        number_of_beamsplitters = ((d - 1) // 2) * d

    thetas_1 = [tf.Variable(0.1) for _ in range(number_of_beamsplitters)]
    phis_1 = [tf.Variable(0.0) for _ in range(d - 1)]

    squeezings = [tf.Variable(0.1) for _ in range(d)]

    thetas_2 = [tf.Variable(0.1) for _ in range(number_of_beamsplitters)]
    phis_2 = [tf.Variable(0.0) for _ in range(d - 1)]

    displacements = [tf.Variable(0.1) for _ in range(d)]

    kappas = [tf.Variable(0.1) for _ in range(d)]

    return {
        "d": d,
        "thetas_1": thetas_1,
        "phis_1": phis_1,
        "squeezings": squeezings,
        "thetas_2": thetas_2,
        "phis_2": phis_2,
        "displacements": displacements,
        "kappas": kappas,
    }


input = 0.01
d = 8
cutoff = 8

target_state_vector = np.zeros(cutoff_cardinality(cutoff=cutoff, d=d), dtype=complex)
target_state_vector[1] = 1.0
target_state = tf.constant(target_state_vector, dtype=tf.complex128)

simulator = pq.TensorflowPureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

parameters = create_layer_parameters(d)

with tf.GradientTape() as tape:

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(r=input)

        i = 0
        for col in range(d):
            if col % 2 == 0:
                for mode in range(0, d - 1, 2):
                    modes = (mode, mode + 1)
                    pq.Q(*modes) | pq.Beamsplitter(parameters["thetas_1"][i], phi=0.0)
                    i += 1

            if col % 2 == 1:
                for mode in range(1, d - 1, 2):
                    modes = (mode, mode + 1)
                    pq.Q(*modes) | pq.Beamsplitter(parameters["thetas_1"][i], phi=0.0)
                    i += 1

        for i in range(d - 1):
            pq.Q(i) | pq.Phaseshifter(parameters["phis_1"][i])

        pq.Q(all) | pq.Squeezing(parameters["squeezings"])

        i = 0
        for col in range(d):
            if col % 2 == 0:
                for mode in range(0, d - 1, 2):
                    modes = (mode, mode + 1)
                    pq.Q(*modes) | pq.Beamsplitter(parameters["thetas_2"][i], phi=0.0)
                    i += 1

            if col % 2 == 1:
                for mode in range(1, d - 1, 2):
                    modes = (mode, mode + 1)
                    pq.Q(*modes) | pq.Beamsplitter(parameters["thetas_2"][i], phi=0.0)
                    i += 1

        for i in range(d - 1):
            pq.Q(i) | pq.Phaseshifter(parameters["phis_2"][i])

        pq.Q(all) | pq.Displacement(r=parameters["displacements"])
        pq.Q(all) | pq.Kerr(parameters["kappas"])

    start_time = time.time()
    state = simulator.execute(program).state
    print("EXECUTION TIME: ", time.time() - start_time)

    state_vector = state._state_vector

    cost = tf.reduce_sum(tf.abs(target_state_vector - state_vector))


profiler_options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
)

# tf.profiler.experimental.start("logdir", options=profiler_options)

flattened_parameters = (
    parameters["thetas_1"]
    + parameters["phis_1"]
    + parameters["squeezings"]
    + parameters["thetas_2"]
    + parameters["phis_2"]
    + parameters["displacements"]
    + parameters["kappas"]
)

start_time = time.time()
gradient = tape.gradient(cost, flattened_parameters)

print("JACOBIAN CALCULATION TIME:", time.time() - start_time)
print("JACOBIAN SHAPE:", [g.numpy() for g in gradient])

# tf.profiler.experimental.stop()
