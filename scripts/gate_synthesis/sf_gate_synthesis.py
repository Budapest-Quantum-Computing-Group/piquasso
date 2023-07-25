import numpy as np

import piquasso as pq

#import strawberryfields as sf
#from strawberryfields.ops import *
#from strawberryfields.utils import operation

# Cutoff dimension
cutoff = 10

# gate cutoff
gate_cutoff = 4

# Number of layers
depth = 15

# Number of steps in optimization routine performing gradient descent
reps = 200

# Learning rate
lr = 0.025

# Standard deviation of initial parameters
passive_sd = 0.1
active_sd = 0.001

import tensorflow as tf

# set the random seed
tf.random.set_seed(42)
np.random.seed(42)

# squeeze gate
sq_r = tf.random.normal(shape=[depth], stddev=active_sd)
sq_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# displacement gate
d_r = tf.random.normal(shape=[depth], stddev=active_sd)
d_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# rotation gates
r1 = tf.random.normal(shape=[depth], stddev=passive_sd)
r2 = tf.random.normal(shape=[depth], stddev=passive_sd)

# kerr gate
kappa = tf.random.normal(shape=[depth], stddev=active_sd)


weights = tf.convert_to_tensor([r1, sq_r, sq_phi, r2, d_r, d_phi, kappa], dtype=np.float64)
weights = tf.Variable(tf.transpose(weights))

sf_params = []
names = ["r1", "sq_r", "sq_phi", "r2", "d_r", "d_phi", "kappa"]

sf_params = np.array(sf_params)
print(sf_params.shape)

calculator = pq._backends.tensorflow.calculator.TensorflowCalculator()
tnp = calculator.np # gradient had None values
config = pq.Config(dtype=np.complex128)
fock_space = pq._math.fock.FockSpace(d=1, cutoff=cutoff, calculator=calculator, config=config)


def get_single_mode_kerr_matrix(xi: float):
    coefficients = [tnp.exp(1j * xi * n**2) for n in range(cutoff)]
    return tnp.diag(coefficients)


def get_single_mode_phase_shift_matrix(phi: float):
    coefficients = [tnp.exp(1j * phi * n) for n in range(cutoff)]
    return tnp.diag(coefficients)


from strawberryfields.utils import random_interferometer
# define unitary up to gate_cutoff
random_unitary = random_interferometer(gate_cutoff)

# extend unitary up to cutoff
target_unitary = np.identity(cutoff, dtype=np.complex128)
target_unitary[:gate_cutoff, :gate_cutoff] = random_unitary

target_kets = np.array([target_unitary[:, i] for i in np.arange(gate_cutoff)])
target_kets = tf.constant(target_kets, dtype=tf.complex128)

def approx_unitary_with_cvnn(layer_params, number_of_layers):
    result_matrix = tnp.identity(cutoff, dtype=np.complex128)

    for j in range(number_of_layers):
        phase_shifter_1_matrix = get_single_mode_phase_shift_matrix(layer_params[j, 0])
        squeezing_matrix = fock_space.get_single_mode_squeezing_operator(r=layer_params[j, 1], phi=layer_params[j, 2])
        phase_shifter_2_matrix = get_single_mode_phase_shift_matrix(layer_params[j, 3])
        displacement_matrix = fock_space.get_single_mode_displacement_operator(r=layer_params[j, 4], phi=layer_params[j, 5])
        kerr_matrix = get_single_mode_kerr_matrix(layer_params[j, 6])
        result_matrix = kerr_matrix @ displacement_matrix @ phase_shifter_2_matrix @ squeezing_matrix @ phase_shifter_1_matrix @ result_matrix
        # result_matrix = result_matrix @ phase_shifter_1_matrix @ squeezing_matrix @ phase_shifter_2_matrix @ displacement_matrix @ kerr_matrix

    return result_matrix


def cost(weights):

    # Run engine
    unitary = approx_unitary_with_cvnn(weights, depth)

    ket = unitary[:, :gate_cutoff].T
    # overlaps
    overlaps = tf.math.real(tf.einsum('bi,bi->b', tf.math.conj(target_kets), ket))
    # Objective function to minimize
    cost = tf.abs(tf.reduce_sum(overlaps - 1))

    return cost, overlaps, ket

# Using Adam algorithm for optimization
opt = tf.keras.optimizers.Adam(learning_rate=lr)

overlap_progress = []
cost_progress = []

# Run optimization
for i in range(reps):

    # one repetition of the optimization
    with tf.GradientTape() as tape:
        loss, overlaps_val, ket_val = cost(weights)

    # calculate the mean overlap
    # This gives us an idea of how the optimization is progressing
    mean_overlap_val = np.mean(overlaps_val)

    # store cost at each step
    cost_progress.append(loss)
    overlap_progress.append(overlaps_val)

    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        # print progress
        print("Rep: {} Cost: {:.4f} Mean overlap: {:.4f}".format(i, loss, mean_overlap_val))


from matplotlib import pyplot as plt
# %matplotlib inline
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.style.use('default')

plt.plot(cost_progress)
plt.ylabel('Cost')
plt.xlabel('Step')
plt.show()


######################################################################
# We can use matrix plots to plot the real and imaginary components of the
# target unitary :math:`V` and learnt unitary :math:`U`.
#

learnt_unitary = ket_val.numpy().T[:gate_cutoff, :gate_cutoff]
target_unitary = target_unitary[:gate_cutoff, :gate_cutoff]

fig, ax = plt.subplots(1, 4, figsize=(7, 4))
ax[0].matshow(target_unitary.real, cmap=plt.get_cmap('Reds'))
ax[1].matshow(target_unitary.imag, cmap=plt.get_cmap('Greens'))
ax[2].matshow(learnt_unitary.real, cmap=plt.get_cmap('Reds'))
ax[3].matshow(learnt_unitary.imag, cmap=plt.get_cmap('Greens'))

ax[0].set_xlabel(r'$\mathrm{Re}(V)$')
ax[1].set_xlabel(r'$\mathrm{Im}(V)$')
ax[2].set_xlabel(r'$\mathrm{Re}(U)$')
ax[3].set_xlabel(r'$\mathrm{Im}(U)$')
fig.show()


######################################################################
# Process fidelity
# ----------------
#
# The process fidelity between the two unitaries is defined by
#
# .. math:: F_e  = \left| \left\langle \Psi(V) \mid \Psi(U)\right\rangle\right|^2
#
# where:
#
# -  :math:`\left|\Psi(V)\right\rangle` is the action of :math:`V` on one
#    half of a maximally entangled state :math:`\left|\phi\right\rangle`:
#
# .. math:: \left|\Psi(V)\right\rangle = (I\otimes V)\left|\phi\right\rangle,
#
# -  :math:`V` is the target unitary,
# -  :math:`U` the learnt unitary.
#

I = np.identity(gate_cutoff)
phi = I.flatten()/np.sqrt(gate_cutoff)
psiV = np.kron(I, target_unitary) @ phi
psiU = np.kron(I, learnt_unitary) @ phi



######################################################################
# Therefore, after 200 repetitions, the learnt unitary synthesized via a
# variational quantum circuit has the following process fidelity to the target
# unitary:
#

print(np.abs(np.vdot(psiV, psiU))**2)


######################################################################
# Circuit parameters
# ------------------
#
# We can also query the optimal variational circuit parameters
# :math:`\vec{\theta}` that resulted in the learnt unitary. For example,
# to determine the maximum squeezing magnitude in the variational quantum
# circuit:
#

print(np.max(np.abs(weights[:, 0])))

######################################################################
# Further results
# ---------------
#
# After downloading the tutorial, even more refined results can be obtained by
# increasing the number of repetitions (``reps``), changing the depth of the
# circuit or altering the gate cutoff!
#

######################################################################
# References
# ----------
#
# 1. Juan Miguel Arrazola, Thomas R. Bromley, Josh Izaac, Casey R. Myers,
#    Kamil Brádler, and Nathan Killoran. Machine learning method for state
#    preparation and gate synthesis on photonic quantum computers. `Quantum
#    Science and Technology, 4
#    024004 <https://iopscience.iop.org/article/10.1088/2058-9565/aaf59e>`__,
#    (2019).
#
# 2. Nathan Killoran, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld,
#    Nicolas Quesada, and Seth Lloyd. Continuous-variable quantum neural networks.
#    `Physical Review Research, 1(3), 033063.
#    <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033063>`__,
#    (2019).
