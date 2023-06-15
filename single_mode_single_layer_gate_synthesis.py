import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import piquasso as pq
# Passive gates on one mode are diagonal

# Basic parameters
d = tf.constant(1)
cutoff = tf.constant(14)
number_of_layers = tf.constant(10)
number_of_params = tf.constant(5)
number_of_steps = tf.constant(1000)
calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
np = calculator.np # gradient had None values
config = pq.Config()
fock_space = pq._math.fock.FockSpace(d=1, cutoff=cutoff, calculator=calculator, config=config)

def experimental_cost(target_matrix, result_matrix):
    return tf.norm(target_matrix - result_matrix)/tf.norm(result_matrix)

def norm1_cost(target_matrix, result_matrix):
    return tf.cast(tf.reduce_sum(tf.math.abs(target_matrix - result_matrix)), dtype=np.float32)

def identity_cost(target_matrix, result_matrix):
    cost = 0
    for i in range(cutoff):
        cost += tf.math.abs((target_matrix @ result_matrix)[i, i] - 1)
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

cubic_phase_param = tf.constant(0.1)
cubic_phase_matrix = fock_space.get_single_mode_cubic_phase_operator(gamma=cubic_phase_param, hbar=config.hbar, calculator=calculator)

layer_params = init_single_layer_parameters_single_mode(layer_num=number_of_layers)
opt = tf.keras.optimizers.Adam(learning_rate=tf.constant(0.001))
for i in tf.range(number_of_steps):
    with tf.GradientTape(persistent=False) as tape:
        result_matrix = np.identity(cutoff)
        for j in tf.range(number_of_layers):
            phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(layer_params[0+j*number_of_params])
            squeezing_matrix = fock_space.get_single_mode_squeezing_operator(r=layer_params[1+j*number_of_params], phi=0)
            phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(layer_params[2+j*number_of_params])
            displacement_matrix = fock_space.get_single_mode_displacement_operator(r=layer_params[3+j*number_of_params], phi=0)
            kerr_matrix = get_single_mode_kerr_matrix(layer_params[4+j*number_of_params])
            #kerr_matrix = fock_space.get_single_mode_cubic_phase_operator(gamma=layer_params[j*number_of_params], hbar=config.hbar, calculator=calculator)

            result_matrix = result_matrix @ phase_shifter_1_matrix @ squeezing_matrix @ phase_shifter_2_matrix @ displacement_matrix @ kerr_matrix
            #result_matrix = kerr_matrix
        cost = norm1_cost(target_matrix=cubic_phase_matrix, result_matrix=result_matrix)
    # breakpoint()
    print("Cost: " + str(cost.numpy()) + " at step: " + str(i))
    # Perform gradient descent
    gradient = tape.gradient(cost, layer_params)
    for j in tf.range(len(gradient)): # SOME elemnts are float64, others are not
        gradient[j] = tf.cast(gradient[j], dtype=np.float32)
    # breakpoint()
    opt.apply_gradients(zip(gradient, layer_params))
    # Casting back to tf.Variable is important
    # layer_params = [tf.Variable(layer_params[j] - learning_rate*gradient[j]) for j in range(len(layer_params))]
