import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import piquasso as pq
import strawberryfields as sf

import numpy as np
import tensorflow as tf


MIN_MODE = 1
MAX_MODE = 6

MIN_CUTOFF = 6
MAX_CUTOFF = 8

d = MIN_MODE
cutoff = MIN_CUTOFF
extra_pq_cutoff = 1

r = 0.05
alpha = 0.01

s = 0.01


def pq_ps():
    with pq.Program() as plus_program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(alpha=alpha)
        pq.Q(all) | pq.Squeezing(r + s)

    with pq.Program() as minus_program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(alpha=alpha)
        pq.Q(all) | pq.Squeezing(r - s)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff + extra_pq_cutoff)
    )

    start_time = time.time()

    plus_state = simulator.execute(plus_program).state
    minus_state = simulator.execute(minus_program).state

    plus_mean = plus_state.mean_position(mode=0)
    minus_mean = minus_state.mean_position(mode=0)

    grad = (plus_mean - minus_mean) / (2 * np.sinh(s))

    return grad, time.time() - start_time


def pq_tf():
    alpha_ = tf.Variable(alpha)
    r_ = tf.Variable(r)

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(alpha=alpha_)
        pq.Q(all) | pq.Squeezing(r_)

    simulator = pq.TensorflowPureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

    start_time = time.time()
    with tf.GradientTape() as tape:
        state = simulator.execute(program).state
        mean = state.mean_position(mode=0)

    grad = tape.jacobian(mean, [r_, alpha_])[0].numpy()

    return grad, time.time() - start_time


def sf_ps():
    plus_program = sf.Program(d)
    minus_program = sf.Program(d)

    plus_engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
    minus_engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

    with plus_program.context as q:
        for i in range(d):
            sf.ops.Dgate(alpha) | q[i]
            sf.ops.Sgate(r + s) | q[i]

    with minus_program.context as q:
        for i in range(d):
            sf.ops.Dgate(alpha) | q[i]
            sf.ops.Sgate(r - s) | q[i]

    start_time = time.time()

    minus_state = minus_engine.run(minus_program).state
    plus_state = plus_engine.run(plus_program).state
    plus_mean, _ = plus_state.quad_expectation(mode=0)
    minus_mean, _ = minus_state.quad_expectation(mode=0)

    grad = (plus_mean - minus_mean) / (2 * np.sinh(s))

    return grad, time.time() - start_time


def sf_tf():
    alpha_ = tf.Variable(alpha)
    r_ = tf.Variable(r)
    program = sf.Program(d)

    mapping = {}
    r_param = program.params("r")
    alpha_param = program.params("alpha")
    mapping["r"] = r_
    mapping["alpha"] = alpha_

    engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})

    with program.context as q:
        for i in range(d):
            sf.ops.Dgate(alpha_param) | q[i]
            sf.ops.Sgate(r_param) | q[i]

    start_time = time.time()

    with tf.GradientTape() as tape:
        result = engine.run(program, args=mapping)
        state = result.state
        mean, _ = state.quad_expectation(mode=0)

    grad = tape.gradient(mean, [r_, alpha_])[0].numpy()

    return grad, time.time() - start_time


pq_ps_avg_et = 0
pq_tf_avg_et = 0
sf_ps_avg_et = 0
sf_tf_avg_et = 0

ref_grad_value = -0.0190246
r_tol = 1e-6

runs = (MAX_MODE - MIN_MODE) * (MAX_CUTOFF - MIN_CUTOFF)

for i in range(MIN_MODE, MAX_MODE):
    for j in range(MIN_CUTOFF, MAX_CUTOFF):
        pq_tf_grad, pq_tf_time = pq_tf()

for i in range(MIN_MODE, MAX_MODE):
    for j in range(MIN_CUTOFF, MAX_CUTOFF):
        d = i
        cutoff = j
        print("MODE_NUM:", d)
        print("CUTOFF_NUMS:", cutoff, cutoff + extra_pq_cutoff)

        sf_tf_grad, sf_tf_time = sf_tf()
        # pq_ps_grad, pq_ps_time = pq_ps()
        # sf_ps_grad, sf_ps_time = sf_ps()

        # print("PQ_TF EXEC TIME:",pq_tf_time)
        # print("PQ_PS EXEC TIME:",pq_ps_time)
        # print("SF_TF EXEC TIME:",sf_tf_time)
        # print("SF_PS EXEC TIME:",sf_ps_time)

        # pq_ps_avg_et += pq_ps_time
        # pq_tf_avg_et += pq_tf_time
        # sf_ps_avg_et += sf_ps_time
        # sf_tf_avg_et += sf_tf_time

        # print("PQ_PS:",np.isclose(pq_ps_grad, ref_grad_value))
        # print("PQ_PS:",np.isclose(pq_tf_grad, ref_grad_value))
        # print("PQ_PS:",np.isclose(sf_ps_grad, ref_grad_value))
        # print("PQ_TF:",np.isclose(sf_tf_grad, ref_grad_value))

        # print("PQ_PS GRAD:",pq_ps_grad)
        # print("PQ_TF GRAD:",pq_tf_grad)
        # print("SF_TF GRAD:",sf_tf_grad)
        # print("SF_PS GRAD:",sf_ps_grad)

print("AVGTIMES:")

# print("PQ_PS:",pq_ps_avg_et/runs)
print("PQ_TF:", pq_tf_avg_et / runs)
# print("SF_PS:",sf_ps_avg_et/runs)
print("SF_TF:", sf_tf_avg_et / runs)
