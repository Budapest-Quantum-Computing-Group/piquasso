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
import tensorflow as tf
with tf.device("/CPU:0"):
    import time

    ### Initialize hyperparameters ###
    from param_config import *
    import neural_network, cvnn_approximation, persistence
    from unitary_generation import random_generator, data_cube_generator


    ### Unitary Data Generation ###
    def generate_unitary_datacube(
        degree=degree,
    ):

        list_of_degree_combinations = []
        list_of_degree_coeffs = []

        for i in range(degree+1):
            (combinations, coeffs) = \
                data_cube_generator.get_hamiltonian_terms_of_degree(i)

            list_of_degree_combinations.append(combinations)
            list_of_degree_coeffs.append(coeffs)

        all_combinations, all_coeffs = \
            data_cube_generator.add_up_all_combinations_tf(list_of_degree_combinations, list_of_degree_coeffs)

        map_result = tf.constant(tf.vectorized_map(data_cube_generator.calculate_unitary_from_hamiltonian_tf, all_combinations))

        return tf.transpose(map_result, perm=(0, 2, 1)), all_coeffs # transpose for later usage in cvnn scripts


    def generate_unitary_datapacks(
        degree=degree,
        number_of_unitaries=number_of_unitaries,
        number_of_datapacks=number_of_datapacks,
    ):
        datapacks = []

        for _ in range(number_of_datapacks):

            (random_kets, random_coeffs)\
                = random_generator.generate_number_of_random_unitaries(
                    degree=degree, amount=number_of_unitaries
            )
            datapacks.append((random_kets, random_coeffs))

        return datapacks


    ### Save the results of the CVNN approximations ###
    def save_datapack_result(data_to_save):
        data_manager = persistence.Persistence(cvnn_path=cvnn_path, nn_path=nn_path)

        data_manager.save_cvnn_data(general_cvnn_info, data_to_save)


    ### Neural Network Testing ###
    ## TODO: This should be the loss function of the new approach
    def test_neural_network_prediciton():
        model, x_train, y_train =\
            neural_network.train_model(
                path=nn_path,
                epochs=number_of_nn_epochs,
                batch_size=128,
                loss=nn_loss,
                optimizer=nn_optimizer
                )
        sum_cost = 0
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
        predicted_cvnn_weights = model.predict(x_train)
        for i in range(len(x_train)):
            unitary, _ = random_generator.generate_random_unitary(degree, x_train[i])
            cvnn_unitary = cvnn_approximator._calculate_unitary_with_cvnn_default(predicted_cvnn_weights[i])
            ket = tf.transpose(cvnn_unitary[:, : cvnn_approximator._gate_cutoff])
            overlaps = tf.math.real(tf.einsum("bi,bi->b", tf.math.conj(unitary), ket))
            cost = tf.abs(tf.reduce_sum(overlaps - 1))
            print("Cost:", cost)
            sum_cost += cost
        print("Avg Cost:", sum_cost/len(x_train))


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


    #### CVNN Data Generation ####
    def cvnn_data_generation_from_random_unitaries():
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

    def cvnn_data_generation_from_data_cube():
        print("Generating data cube...")
        datacube = generate_unitary_datacube()
        # datapack = generate_unitary_datapacks(6, 10, 2)
        print("Data generation finished")
        print("Begin processing datacube")
        start_time = time.time()
        data_to_save = approximate_datapack_with_cvnn(datacube)
        {"Datacube process finished in:", time.time() - start_time}
        print("Saving data")
        save_datapack_result(data_to_save)
        print("Data saved")

    if __name__ == "__main__":
        cvnn_data_generation_from_data_cube()
