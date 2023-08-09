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
from tensorflow.python.keras.layers import Dense, Reshape
import json

np.set_printoptions(suppress=True, linewidth=200)


def read_data(path: str):

    with open(path) as json_file:
        json_data = json.load(json_file)
    general_info = json_data["general_info"]
    data = json_data["data"]

    target_coeffs = data["target_coeffs"]
    cvnn_weights = data["cvnn_weights"]

    target_coeffs = eval(
        "np.array(" + target_coeffs + ")"
    )  # (unitary_amount, order_dependend)
    cvnn_weights = eval(
        "np.array(" + cvnn_weights + ")"
    )  # (unitary_amount, layer_amount, gate_amount)

    return target_coeffs, cvnn_weights, general_info


def matching_general_info(info1: dict, info2: dict):
    return (
        info1["number_of_layers"] == info2["number_of_layers"]
        and info1["hamiltonian_order"] == info2["hamiltonian_order"]
        and info1["gate_cutoff"] == info2["gate_cutoff"]
    )


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
            breakpoint()
            return None

    if aslist:
        return target_coeffs_list, cvnn_weights_list

    target_coeffs_array = np.vstack(target_coeffs_list)
    cvnn_weights_array = np.vstack(cvnn_weights_list)

    return target_coeffs_array, cvnn_weights_array


def split_data(
    data, train_ratio: float = 0.6, val_ratio: float = 0.3, test_ratio: float = 0.1
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


def create_model(input_size: int, output_size: int, loss="mse", optimizer="adam"):

    model = Sequential()
    output_product = output_size[0] * output_size[1]
    model.add(Dense(units=16 * input_size, activation="relu", input_dim=input_size))
    model.add(Dense(units=32 * input_size, activation="relu"))
    model.add(Dense(units=64 * input_size, activation="relu"))
    model.add(Dense(units=32 * input_size, activation="relu"))
    model.add(Dense(units=16 * input_size, activation="relu"))
    model.add(Dense(units=output_product, activation="linear"))
    model.add(Reshape(output_size))
    model.compile(loss=loss, optimizer=optimizer, metrics="accuracy")

    return model


def train_model(path, epochs, batch_size):

    x_data, y_data = read_all_data("./scripts/gate_synthesis/cvnn_approximations/")
    # x_train, y_train = read_all_data( "./scripts/gate_synthesis/cvnn_approximations/")
    x_train, x_val, x_test = split_data(x_data)
    y_train, y_val, y_test = split_data(y_data)

    model = create_model(x_train.shape[1], y_train[0].shape)
    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    print("FINISHED")


def test_model(path, epochs, batch_size):

    x_train_list, y_train_list = read_all_data(
        "./scripts/gate_synthesis/cvnn_approximations/", aslist=True
    )

    for i in range(len(x_train_list)):
        model = create_model(x_train_list[i].shape[1], y_train_list[i][0].shape)

        print(model.summary())
        history = model.fit(
            x_train_list[i], y_train_list[i], epochs=epochs, batch_size=batch_size
        )

        breakpoint()

    print("FINISHED")


if __name__ == "__main__":
    # TODO: Generate even more data.
    train_model(
        "./scripts/gate_synthesis/cvnn_approximations/", epochs=200, batch_size=32
    )
    # test_model("./scripts/gate_synthesis/cvnn_approximations/", epochs=200, batch_size=32)
