import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import piquasso as pq
import normal_ordering
# Passive gates on one mode are diagonal

# Basic parameters
d = tf.constant(1)
cutoff = tf.constant(20)
number_of_layers = tf.constant(200)
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

cubic_phase_param = tf.constant(0.1)
cubic_phase_matrix = fock_space.get_single_mode_cubic_phase_operator(gamma=cubic_phase_param, hbar=config.hbar, calculator=calculator)

order = 7  # What's a good order? # Maximal order for neural layer
batch_size = 10

unitaries = []
for i in range(batch_size):
    unitaries.append(normal_ordering.generate_unitary(order, cutoff))
    # Koefficienseket kell átadni a hálónak

layer_params = init_single_layer_parameters_single_mode(layer_num=number_of_layers)  # After neural layer try non-cvnn
opt = tf.keras.optimizers.Adam(learning_rate=tf.constant(0.001))
min_cost = sys.maxsize - 1  # Although Python3 has no upper bound for integers
sum_cost = 0

for i in range(number_of_steps):

    with tf.GradientTape(persistent=False) as tape:
        result_matrix = np.identity(cutoff)

        for j in range(number_of_layers):
            phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(layer_params[0+j*number_of_params])
            squeezing_matrix = fock_space.get_single_mode_squeezing_operator(r=layer_params[1+j*number_of_params], phi=0)
            phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(layer_params[2+j*number_of_params])
            displacement_matrix = fock_space.get_single_mode_displacement_operator(r=layer_params[3+j*number_of_params], phi=0)
            kerr_matrix = get_single_mode_kerr_matrix(layer_params[4+j*number_of_params])

            result_matrix = result_matrix @ phase_shifter_1_matrix @ squeezing_matrix @ phase_shifter_2_matrix @ displacement_matrix @ kerr_matrix

        cost = 0
        for k in range(batch_size):
            cost += identity_cost(target_matrix=unitaries[k], result_matrix=result_matrix)
            # sum cost -> multiple costs
        if cost < min_cost:
            min_cost = cost
        sum_cost += cost

    print("Cost: " + str(cost.numpy()) + " at step: " + str(i))
    # Perform gradient descent
    gradient = tape.gradient(cost, layer_params)
    for j in tf.range(len(gradient)): # SOME elemnts are float64, others are not
        gradient[j] = tf.cast(gradient[j], dtype=np.float32)

    opt.apply_gradients(zip(gradient, layer_params))
    # Casting back to tf.Variable is important
    # layer_params = [tf.Variable(layer_params[j] - learning_rate*gradient[j]) for j in range(len(layer_params))]
print("FINISHED")
print("Min. cost:", min_cost)
print("Avg. cost:", sum_cost/number_of_steps)
