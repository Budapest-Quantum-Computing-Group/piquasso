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

### Initialize hyperparameters ###
np.set_printoptions(suppress=True, linewidth=200, threshold=sys.maxsize)

## Photonic Quantum Computing ##
number_of_modes = 1
cutoff = 12
dtype = np.complex128

## Unitary ##
degree = 5  # of Hamiltonian
number_of_unitaries = 2
gate_cutoff = 4
seed = None

## CVNN Parameters ##
isProfiling = False
number_of_datapacks = 1
number_of_cvnn_layers = 15
number_of_cvnn_steps = 250
cvnn_tolerance = 0.05
cvnn_learning_rate = 0.05
cvnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cvnn_learning_rate)
passive_sd = 0.1
active_sd = 0.001

## Neural network ##
number_of_nn_steps = 400
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
    for i in range(number_of_datapacks):
        print("Begin processing datapack {}".format(i + 1))
        data_to_save = approximate_datapack_with_cvnn(datapacks[i])
        save_datapack_result(data_to_save)


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
        isProfiling=isProfiling,
        seed=seed,
    )

    sum_time = 0
    for i in range(number_of_unitaries):
        start_time = time.time()
        cvnn_approximator.benchmark_gradient(random_kets[i])
        sum_time += time.time() - start_time

    print(
        "Average runtime over {} unitaries: {}".format(
            number_of_unitaries, sum_time / number_of_unitaries
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
    # benchmark_cvnn()
