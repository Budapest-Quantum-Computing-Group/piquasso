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
with tf.device("/CPU:0"):
    import time

    ### Initialize hyperparameters ###
    from param_config import *
    import neural_network, cvnn_approximation, persistence
    from unitary_generation import random_generator, data_cube_generator


 #### Benchmark ####
    def benchmark_cvnn():
        seed = 42

        random_kets, _ = random_generator.generate_number_of_random_unitaries(
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

        random_kets, _ = random_generator.generate_number_of_random_unitaries(
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

if __name__ == "__main__":
    benchmark_cvnn()