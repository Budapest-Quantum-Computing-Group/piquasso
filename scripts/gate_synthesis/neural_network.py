import numpy as np
import piquasso as pq
import os
import normal_ordering

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
from sklearn.metrics import accuracy_score


d = 1
cutoff = 20
number_of_layers = 200
number_of_params = 5

calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
np = calculator.np  # gradient had None values
config = pq.Config()
fock_space = pq._math.fock.FockSpace(
    d=1, cutoff=cutoff, calculator=calculator, config=config
)


def get_single_mode_kerr_matrix(xi: float):
    coefficients = [np.exp(1j * xi * n**2) for n in range(cutoff)]
    return np.diag(coefficients)


def get_single_mode_phase_shift_matrix(phi: float):
    coefficients = [np.exp(1j * phi) for _ in range(cutoff)]
    return np.diag(coefficients)


def generate_data(amount: int, order: int, seed=None):
    data = []

    for i in range(amount):
        data.append(
            normal_ordering.generate_all_random_normal_polynomial_coeffs(order, seed[i])
        )

    return data


def identity_cost(target_matrix, result_matrix):
    """
    Eq. 9 in https://arxiv.org/pdf/1807.10781.pdf
    """
    cost = 0
    self_multiplication = target_matrix @ np.conj(result_matrix).T

    for i in range(cutoff):
        cost += tf.math.abs((self_multiplication)[i, i] - 1)

    return cost / cutoff


def cvnn_loss(cvnn_params, unitary):
    result_matrix = np.identity(cutoff)

    for j in range(number_of_layers):
        phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(
            cvnn_params[0 + j * number_of_params]
        )
        squeezing_matrix = fock_space.get_single_mode_squeezing_operator(
            r=cvnn_params[1 + j * number_of_params], phi=0
        )
        phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(
            cvnn_params[2 + j * number_of_params]
        )
        displacement_matrix = fock_space.get_single_mode_displacement_operator(
            r=cvnn_params[3 + j * number_of_params], phi=0
        )
        kerr_matrix = get_single_mode_kerr_matrix(cvnn_params[4 + j * number_of_params])

        result_matrix = (
            result_matrix
            @ phase_shifter_1_matrix
            @ squeezing_matrix
            @ phase_shifter_2_matrix
            @ displacement_matrix
            @ kerr_matrix
        )

    cost = identity_cost(target_matrix=unitary, result_matrix=result_matrix)

    return cost


def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(units=32, activation="relu", input_dim=input_size))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=output_size, activation="linear"))
    model.compile(loss=cvnn_loss, optimizer="sgd", metrics="accuracy")

    return model


if __name__ == "__main__":
    order = 5
    data_amount = 5
    seeds = [10, 11, 12, 13, 14]
    x_train = generate_data(data_amount, order, seeds)
    coeff_amount = len(x_train[0])

    print(x_train)
    model = create_model(len(x_train[0]), 5)
    model.summary()
    unitaries = []

    for i in range(data_amount):
        unitaries.append(normal_ordering.generate_unitary(order, cutoff, seeds[i]))

    breakpoint()
    model.fit(x_train, unitaries, epochs=200, batch_size=16)
