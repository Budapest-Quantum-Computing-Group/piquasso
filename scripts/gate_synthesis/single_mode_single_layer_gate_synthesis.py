import numpy as np
import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import piquasso as pq
import normal_ordering
import neural_network
from random import randint
# Passive gates on one mode are diagonal

# Basic parameters
d = 1
cutoff = 20

calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
real_numpy = np
np = calculator.np # gradient had None values
config = pq.Config()
fock_space = pq._math.fock.FockSpace(d=1, cutoff=cutoff, calculator=calculator, config=config)

def experimental_cost(target_matrix, result_matrix):
    return tf.norm(target_matrix - result_matrix)/tf.norm(result_matrix)

def norm1_cost(target_matrix, result_matrix):
    return tf.cast(tf.reduce_sum(tf.math.abs(target_matrix - result_matrix)), dtype=np.float32)

def identity_cost(target_matrix, result_matrix):
    """
    Eq. 9 in https://arxiv.org/pdf/1807.10781.pdf
    """
    cost = 0
    self_multiplication = target_matrix @ np.conj(result_matrix).T

    for i in range(cutoff):
        cost += tf.math.abs((self_multiplication)[i, i] - 1)

    return cost / cutoff

# Functions for matrices not directly avaiable via Piquasso API
def get_single_mode_kerr_matrix(xi: float):
    coefficients = [np.exp(1j*xi*n**2) for n in range(cutoff)]
    return np.diag(coefficients)

def get_single_mode_phase_shift_matrix(phi: float):
    coefficients = [np.exp(1j*phi) for _ in range(cutoff)]
    return np.diag(coefficients)

# Initializing parameters and target gate based on profile_cvnn_layer.py into a flattened list
def init_single_layer_parameters_single_mode(d: int = 1, layer_num: int = 1):
    params = []
    for i in range(layer_num):
        phase_shifter_1 = [tf.Variable(0.3) for _ in range(d)]
        squeezings = [tf.Variable(0.1) for _ in range(d)]
        phase_shifter_2 = [tf.Variable(0.3) for _ in range(d)]
        displacements = [tf.Variable(0.1) for _ in range(d)]
        kerrs = [tf.Variable(0.1) for _ in range(d)]
        params = params + phase_shifter_1 + squeezings + phase_shifter_2 + displacements + kerrs
        #params = kerrs
    return params
    """
    return {
        "phase_shifter_1": phase_shifter_1,
        "squeezing": squeezings,
        "phase_shifter_2": phase_shifter_2,
        "displacement": displacements,
        "kerr": kerrs
    }
    """


def approximate_hamiltonian_unitary_cvnn(unitary, number_of_steps, number_of_layers, cost_function, optimizer):
    # TODO: Reproduce exactly paper
    cubic_phase_param = tf.constant(0.1)
    cubic_phase_matrix = fock_space.get_single_mode_cubic_phase_operator(gamma=cubic_phase_param, hbar=config.hbar, calculator=calculator)
    unitary = cubic_phase_matrix
    min_cost = sys.maxsize - 1  # Although Python3 has no upper bound for integers
    max_cost = sum_cost = 0
    best_matrix = np.empty((cutoff, cutoff))
    layer_params = init_single_layer_parameters_single_mode(layer_num=number_of_layers)  # After neural layer try non-cvnn

    for i in range(number_of_steps):
        # Get predicitons from model for a random unitary in training set, and use those az layer_params
        # TODO: Tensorboard benchmarking
        with tf.GradientTape(persistent=False) as tape:  # Maybe not necessary
            result_matrix = np.identity(cutoff)

            for j in range(number_of_layers):
                phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(layer_params[0+j*number_of_params])
                squeezing_matrix = fock_space.get_single_mode_squeezing_operator(r=layer_params[1+j*number_of_params], phi=0)
                phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(layer_params[2+j*number_of_params])
                displacement_matrix = fock_space.get_single_mode_displacement_operator(r=layer_params[3+j*number_of_params], phi=0)
                kerr_matrix = get_single_mode_kerr_matrix(layer_params[4+j*number_of_params])
                # Lookup tf.einsum
                result_matrix = result_matrix @ phase_shifter_1_matrix @ squeezing_matrix @ phase_shifter_2_matrix @ displacement_matrix @ kerr_matrix
        # TODO: Tensorboard benchmarking
            # Somehow use the cost to adjust the model.
            cost = cost_function(target_matrix=unitary, result_matrix=result_matrix)

            if cost < min_cost:
                min_cost = cost
                best_matrix = result_matrix
            elif cost > max_cost:
                max_cost = cost
            sum_cost += cost

        print("Cost: " + str(cost.numpy()) + " at step: " + str(i))
        # Perform gradient descent
        gradient = tape.gradient(cost, layer_params)
        for j in tf.range(len(gradient)): # SOME elemnts are float64, others are not
            gradient[j] = tf.cast(gradient[j], dtype=np.float32)

        # Optimizer probably saves best result
        optimizer.apply_gradients(zip(gradient, layer_params))
        # Casting back to tf.Variable is important
        # layer_params = [tf.Variable(layer_params[j] - learning_rate*gradient[j]) for j in range(len(layer_params))]

    mean_cost = sum_cost/number_of_steps
    if type(min_cost) != int:
        min_cost = min_cost.numpy()
    if type(max_cost) != int:
        max_cost = max_cost.numpy()
    if type(mean_cost) != int:
        mean_cost = mean_cost.numpy()
    print("FINISHED")
    print("Min. cost:", min_cost)
    print("Mean cost:", mean_cost)
    print("Max cost:", max_cost)
    layer_params = [float(layer_params[i].numpy()) for i in range(len(layer_params))]

    return {
        "min_cost": min_cost,
        "max_cost": max_cost,
        "mean_cost": mean_cost,
        # NOTE: TypeError: Object of type complex is not JSON serializable
        "cvnn_unitary": real_numpy.array2string(best_matrix.numpy()),  # fromstring()
        "layer_params": layer_params,
    }


if __name__ == "__main__":

    number_of_unitaries = 1
    number_of_params = 5

    # What's a good order? # Maximal order for neural layer
    hamiltonian_order = 4
    unitaries = []
    seeds = [(randint(1, 100), randint(1, 100)) for i in range(number_of_unitaries)]
    unitary_coefficients = neural_network.generate_data(number_of_unitaries, hamiltonian_order, seeds)
    coeff_amount = len(unitary_coefficients[0])

    for main_step in range(1):
        number_of_layers = 25
        number_of_steps = 20000

        for i in range(number_of_unitaries):
            unitaries.append(normal_ordering.generate_unitary(hamiltonian_order, cutoff, seeds[i]))

        result = {
            "min_costs": [],
            "max_costs": [],
            "mean_costs": [],
            "cvnn_unitaries": [],
            "layer_params": []
        }

        for i in range(number_of_unitaries):
            opt = tf.keras.optimizers.Adam(learning_rate=tf.constant(0.001))
            costs = approximate_hamiltonian_unitary_cvnn(unitaries[i], number_of_steps, number_of_layers, identity_cost, opt)
            result["min_costs"].append(costs["min_cost"])
            result["max_costs"].append(costs["max_cost"])
            result["mean_costs"].append(costs["mean_cost"])
            result["cvnn_unitaries"].append(costs["cvnn_unitary"])
            result["layer_params"].append(costs["layer_params"])

        result = {
            "info":{
                "number_of_steps": number_of_steps,
                "number_of_layers": number_of_layers,
                "number_of_unitaries": number_of_unitaries,
                "optimizer": "Adam",
                "cost_function": "Identity",
                "hamiltonian_coeffs": real_numpy.array2string(unitary_coefficients.numpy()),  # fromstring()
            },
            "results:": result,
        }

        json_file = open("scripts/gate_synthesis/cvnn_approximations/reproduce_paper.json", "w")
        json.dump(result, json_file, indent=4)
        json_file.close()
        print("MAIN STEP FINISHED")
