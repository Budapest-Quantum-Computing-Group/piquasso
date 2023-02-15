import pytest

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import piquasso as pq
import strawberryfields as sf

import numpy as np
import tensorflow as tf


pytestmark = pytest.mark.benchmark(
    group="param-shift",
)

MIN_MODE = 2
MAX_MODE = 5

@pytest.fixture
def r():
    return 0.05

@pytest.fixture
def cutoff():
    return 10

@pytest.fixture
def alpha():
    return 0.01

s = r


@pytest.mark.parametrize("d", range(MIN_MODE, MAX_MODE))
def pq_ps_benchmark(benchmark, d, cutoff, r, alpha, s):
    @benchmark
    def func():
        with pq.Program() as plus_program:
            pq.Q(all) | pq.Vacuum()

            pq.Q(all) | pq.Displacement(alpha=alpha)
            pq.Q(all) | pq.Squeezing(r + s)

        with pq.Program() as minus_program:
            pq.Q(all) | pq.Vacuum()

            pq.Q(all) | pq.Displacement(alpha=alpha)
            pq.Q(all) | pq.Squeezing(r - s)

        simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        start_time = time.time()

        plus_state = simulator.execute(plus_program).state
        minus_state = simulator.execute(minus_program).state

        plus_mean, _ = plus_state.quadratures_mean_variance(modes=(0,))
        minus_mean, _ = minus_state.quadratures_mean_variance(modes=(0,))

        grad = (plus_mean - minus_mean)/(2*np.sinh(s))

        return grad, time.time() - start_time


@pytest.mark.parametrize("d", range(MIN_MODE, MAX_MODE))
def sf_ps_benchmark(benchmark, d, cutoff, r, alpha, s):
    @benchmark
    def func():
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

        grad = (plus_mean - minus_mean)/(2*np.sinh(s))

        # return grad, time.time() - start_time


#for i in range(1, 6):
#
#    d = i
#    print("MODE_NUM:", d)
#
#    pq_tf_grad, pq_tf_time = pq_tf()
#    pq_ps_grad, pq_ps_time = pq_ps()
#
#    sf_tf_grad, sf_tf_time = sf_tf()
#    sf_ps_grad, sf_ps_time = sf_ps()
#
#    print("PQ_TF EXEC TIME:",pq_tf_time)
#    print("PQ_PS EXEC TIME:",pq_ps_time)
#    print("SF_TF EXEC TIME:",sf_tf_time)
#    print("SF_PS EXEC TIME:",sf_ps_time)
#
#    print("PQ_PS - PQ_TF ISCLOSE:",np.isclose(pq_ps_grad, pq_tf_grad))
#    print("PQ_PS - SF_PS ISCLOSE:",np.isclose(pq_ps_grad, sf_ps_grad))
#    print("PQ_PS - SF_TF ISCLOSE:",np.isclose(pq_ps_grad, sf_tf_grad))
#    print("PQ_TF - SF_PS ISCLOSE:",np.isclose(pq_tf_grad, sf_ps_grad))
#    print("PQ_TF - SF_TF ISCLOSE:",np.isclose(pq_tf_grad, sf_tf_grad))
#    print("SF_PS - SF_TF ISCLOSE:",np.isclose(sf_ps_grad, sf_tf_grad))
#
#    print("PQ_TF GRAD:",pq_tf_grad)
#    print("PQ_PS GRAD:",pq_ps_grad)
#    print("SF_TF GRAD:",sf_tf_grad)
#    print("SF_PS GRAD:",sf_ps_grad)
