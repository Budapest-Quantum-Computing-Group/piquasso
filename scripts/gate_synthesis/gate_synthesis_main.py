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

import neural_network, cvnn_approximation, persistence, unitary_generation

# If system does not do it automatically
sys.path.append(".")
tf.debugging.disable_traceback_filtering()
### Initialize hyperparameters ###
np.set_printoptions(suppress=True, linewidth=200, threshold=sys.maxsize)
with tf.device("/CPU:0"):
    ## Photonic Quantum Computing ##
    number_of_modes = 1
    cutoff = 12
    dtype = np.complex128

    ## Unitary ##
    degree = 6  # of Hamiltonian
    number_of_unitaries = 300
    gate_cutoff = 4
    seed = None

    ## CVNN Parameters ##
    isProfiling = False
    number_of_datapacks = 50
    number_of_cvnn_layers = 15
    number_of_cvnn_steps = 400
    cvnn_tolerance = 0.025
    cvnn_learning_rate = 0.05
    cvnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cvnn_learning_rate)
    passive_sd = 0.1
    active_sd = 0.001

    ## Neural network ##
    number_of_nn_steps = 1
    nn_learning_rate = 0.05
    nn_optimizer = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)

    ## Persistence ##
    cvnn_path = "scripts/gate_synthesis/cvnn_approximations/"
    nn_path = "scripts/gate_synthesis/neural_netwokr_checkpoints/"
    general_cvnn_info = {
        "cutoff": cutoff,
        "mode_amount": number_of_modes,
        "gate_cutoff": gate_cutoff,
        "number_of_layers": number_of_cvnn_layers,
        "number_of_steps": number_of_cvnn_steps,
        "number_of_unitaries": number_of_unitaries,
        "learnin_rate": cvnn_learning_rate,
        "active_sd": active_sd,
        "passive_sd": passive_sd,
        "seed": seed,
        "hamiltonian_degree": degree,
        "optimizer": cvnn_optimizer.__str__(),
        "tolerance": cvnn_tolerance,
    }

    general_nn_info = {
        "number_of_steps": number_of_nn_steps,
        "learnin_rate": nn_learning_rate,
        "optimizer": nn_optimizer.__str__(),
    }

    #### Main ####
    def main_task():
        datapacks = generate_unitary_datapacks()
        sum_time = 0
        for i in range(number_of_datapacks):
            print("Begin processing datapack {}".format(i + 1))
            start_time = time.time()
            data_to_save = approximate_datapack_with_cvnn(datapacks[i])
            computation_time = time.time() - start_time
            sum_time += computation_time
            {"Datapack process finished in:", computation_time}
            save_datapack_result(data_to_save)

        print("Tasks finished. Average time on a single unitary:", sum_time/(number_of_datapacks * number_of_unitaries))


    #### Benchmark ####
    def benchmark_cvnn():
        seed = 42

        unitary_generator = unitary_generation.UnitaryGenerator(
            cutoff=cutoff, gate_cutoff=gate_cutoff, dtype=dtype, seed=seed
        )

        random_kets, _ = unitary_generator.generate_number_of_random_unitaries(
            degree=degree, amount=number_of_unitaries
        )

        cvnn_approximator = cvnn_approximation.CVNNApproximator(
            cutoff=cutoff,
            gate_cutoff=gate_cutoff,
            dtype=dtype,
            number_of_layers=number_of_cvnn_layers,
            number_of_steps=1,
            tolerance=cvnn_tolerance,
            optimizer_learning_rate=cvnn_learning_rate,
            optimizer=cvnn_optimizer,
            passive_sd=passive_sd,
            active_sd=active_sd,
            isProfiling=False,
            seed=seed,
        )

        sum_time = 0
        for i in range(number_of_unitaries):
            if i > 0:
                cvnn_approximator._isProfiling =  True
            times = cvnn_approximator.benchmark_gradient(random_kets[i])
            sum_time += times[0] + times[1]

        print(
            "Average runtime over {} unitaries: {}".format(
                number_of_unitaries, sum_time / number_of_unitaries
            )
        )

    def benchmark_two_cvnns():

        seed = 42

        unitary_generator = unitary_generation.UnitaryGenerator(
            cutoff=cutoff, gate_cutoff=gate_cutoff, dtype=dtype, seed=seed
        )

        random_kets, _ = unitary_generator.generate_number_of_random_unitaries(
            degree=degree, amount=number_of_unitaries
        )

        cvnn_approximator = cvnn_approximation.CVNNApproximator(
            cutoff=cutoff,
            gate_cutoff=gate_cutoff,
            dtype=dtype,
            number_of_layers=number_of_cvnn_layers,
            number_of_steps=1,
            tolerance=cvnn_tolerance,
            optimizer_learning_rate=cvnn_learning_rate,
            optimizer=cvnn_optimizer,
            passive_sd=passive_sd,
            active_sd=active_sd,
            isProfiling=False,
            seed=seed,
        )

        sum1_time = 0
        sum2_time = 0
        for i in range(number_of_unitaries):
            if i > 0:
                cvnn_approximator._isProfiling = isProfiling
            times = cvnn_approximator.benchmark_two_gradients(random_kets[i])
            if i > 0:
                sum1_time += times[0] + times[1]
                sum2_time += times[2] + times[3]

        print(
            "Average runtime1 was {} and runtime2 was {} over {} unitaries.".format(
                sum1_time / number_of_unitaries, sum2_time / number_of_unitaries, number_of_unitaries
            )
        )


    #### Subtask functions ####
    ### Generate unitary data ###
    def generate_unitary_datapacks(
        degree=degree,
        number_of_unitaries=number_of_unitaries,
        number_of_datapacks=number_of_datapacks,
    ):
        datapacks = []

        unitary_generator = unitary_generation.UnitaryGenerator(
            cutoff=cutoff, gate_cutoff=gate_cutoff, dtype=dtype, seed=seed
        )

        for _ in range(number_of_datapacks):

            (
                random_kets,
                random_coeffs,
            ) = unitary_generator.generate_number_of_random_unitaries(
                degree=degree, amount=number_of_unitaries
            )

            datapacks.append((random_kets, random_coeffs))

        return datapacks


    ### Approximate with CVNN ###
    def approximate_datapack_with_cvnn(datapack):
        cvnn_approximator = cvnn_approximation.CVNNApproximator(
            cutoff=cutoff,
            gate_cutoff=gate_cutoff,
            dtype=dtype,
            number_of_layers=number_of_cvnn_layers,
            number_of_steps=number_of_cvnn_steps,
            tolerance=cvnn_tolerance,
            optimizer_learning_rate=cvnn_learning_rate,
            optimizer=cvnn_optimizer,
            passive_sd=passive_sd,
            active_sd=active_sd,
            isProfiling=isProfiling,
            seed=seed,
        )
        costs, weights = cvnn_approximator.approximate_kets(datapack[0])
        data_to_save = {
            "target_coeffs": np.array2string(datapack[1], separator=","),
            "cvnn_weights": np.array2string(np.asarray(weights), separator=","),
            "min_cost": costs,
        }

        return data_to_save


    ### Save the results of the CVNN approximations ###
    def save_datapack_result(data_to_save):
        data_manager = persistence.Persistence(cvnn_path=cvnn_path, nn_path=nn_path)

        data_manager.save_cvnn_data(general_cvnn_info, data_to_save)


    if __name__ == "__main__":
        main_task()
        # benchmark_two_cvnns()
        # benchmark_cvnn()
