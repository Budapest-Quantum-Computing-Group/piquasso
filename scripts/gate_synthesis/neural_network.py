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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Reshape, Dropout
from tensorflow.python.keras.optimizers import adam_v2, gradient_descent_v2
import tensorflow.python.keras.regularizers as regularizers
import json

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
np.set_printoptions(suppress=True, linewidth=200)


def read_data(path: str):

    with open(path) as json_file:
        json_data = json.load(json_file)
    general_info = json_data["general_info"]
    data = json_data["data"]

    target_coeffs = data["target_coeffs"]
    cvnn_weights = data["cvnn_weights"]

    min_costs = data["min_cost"]

    print("Mean min cost in ", path, np.mean(min_costs))

    target_coeffs = eval(
        "np.array(" + target_coeffs + ")"
    )  # (unitary_amount, order_dependend)
    cvnn_weights = eval(
        "np.array(" + cvnn_weights + ")"
    )  # (unitary_amount, layer_amount, gate_amount)

    return target_coeffs, cvnn_weights, general_info


def matching_general_info(info1: dict, info2: dict):
    matching_number_of_layers = info1["number_of_layers"] == info2["number_of_layers"]
    matching_gate_cutoff = info1["gate_cutoff"] == info2["gate_cutoff"]
    if "hamiltonian_order" in info1 and "hamiltonian_order" in info2:
        matching_degree = info1["hamiltonian_order"] == info2["hamiltonian_order"]
    if "hamiltonian_order" in info1 and "hamiltonian_degree" in info2:
        matching_degree = info1["hamiltonian_order"] == info2["hamiltonian_degree"]
    if "hamiltonian_degree" in info1 and "hamiltonian_order" in info2:
        matching_degree = info1["hamiltonian_degree"] == info2["hamiltonian_order"]
    elif "hamiltonian_degree" in info1 and "hamiltonian_degree" in info2:
        matching_degree = info1["hamiltonian_degree"] == info2["hamiltonian_degree"]

    return matching_number_of_layers and matching_degree and matching_gate_cutoff


def read_all_data(path: str, aslist: bool = False):

    content_list = os.listdir(path)
    content_list = [
        path + "/" + f
        for f in content_list
        if (os.path.isfile(path + "/" + f) and ".json" in f)
    ]
    target_coeffs_list = []
    cvnn_weights_list = []
    data_amount = 0
    general_info_list = []
    for json_file in content_list:
        target_coeffs, cvnn_weights, general_info = read_data(json_file)
        target_coeffs_list.append(target_coeffs)
        cvnn_weights_list.append(cvnn_weights)
        general_info_list.append(general_info)

        data_amount += target_coeffs.shape[0]

    for i in range(1, len(general_info_list)):
        if not matching_general_info(general_info_list[i - 1], general_info_list[i]):
            return None

    if aslist:
        return target_coeffs_list, cvnn_weights_list
    target_coeffs_array = np.vstack(target_coeffs_list)
    cvnn_weights_array = np.vstack(cvnn_weights_list)

    return target_coeffs_array, cvnn_weights_array


def split_data(
    data, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1
):

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print("Wrong split ratios")
        return None

    if type(data) == list:
        data = np.vstack(data)

    data_amount = data.shape[0]
    train_index = int(np.floor(data_amount * train_ratio))
    val_index = train_index + int(np.floor(data_amount * val_ratio))

    train_data = data[:train_index, :]
    val_data = data[train_index:val_index, :]
    test_data = data[val_index:, :]

    return train_data, val_data, test_data


def create_model(input_size: int, output_size: int, loss="mse", optimizer="sgd"):

    output_product = output_size[0] * output_size[1]
    print("INPUTSIZE:", input_size)
    print("OUTPUTPRODUCT:", output_product)
    print("OUTPUTSIZE:", output_size)
    optimizer = gradient_descent_v2.SGD(learning_rate=0.1)
    kernel_regularizer=regularizers.l2(0.1)
    kernel_regularizer=None
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Dense(units=50 * input_size, activation="relu", input_dim=input_size))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(0.2))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(0.2))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(0.2))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=30 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(0.2))
    model.add(Dense(units=30 * input_size, activation="relu"))
    model.add(Dense(units=50 * input_size, activation="relu", kernel_regularizer=kernel_regularizer))
    model.add(Dense(units=output_product, activation="linear"))
    model.add(Reshape(output_size))
    model.compile(loss=loss, optimizer=optimizer)

    return model


def train_model(path, epochs, batch_size, loss="mse", optimizer="adam"):

    x_data, y_data = read_all_data(path)
    # x_train, y_train = read_all_data(path)
    x_train, x_val, x_test = split_data(x_data)
    y_train, y_val, y_test = split_data(y_data)

    model = create_model(x_train.shape[1], y_train[0].shape, loss=loss, optimizer=optimizer)
    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )
    print("FINISHED")
    return model, x_train, y_train


def test_model(path, epochs, batch_size, loss="mse", optimizer="adam"):

    x_train_list, y_train_list = read_all_data(
        path, aslist=True
    )

    for i in range(len(x_train_list)):
        model = create_model(x_train_list[i].shape[1], y_train_list[i][0].shape, loss, optimizer)
        print(model.summary())
        history = model.fit(
            x_train_list[i], y_train_list[i], epochs=epochs, batch_size=batch_size
        )
        return model, x_train_list[i], y_train_list[i]
        breakpoint()

    print("FINISHED")


if __name__ == "__main__":
    # TODO: Generate even more data.
    epochs = 1000
    batch_size = 128
    train_model(
        "./scripts/gate_synthesis/cvnn_approximations/", epochs=epochs, batch_size=batch_size
    )
    #test_model("./scripts/gate_synthesis/cvnn_approximations/", epochs=epochs, batch_size=batch_size)
