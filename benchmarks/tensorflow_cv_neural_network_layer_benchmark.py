#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

import pytest

import numpy as np

import piquasso as pq
import strawberryfields as sf

import tensorflow as tf

from scipy.stats import unitary_group

pytestmark = pytest.mark.benchmark(
    group="tensorflow-cv-neural-network-layer-comparison",
)


@pytest.fixture
def cutoff():
    return 3


d_tuple = (2,)
x_tuple = tuple([np.random.rand(d) for d in d_tuple])
U1_tuple = tuple([unitary_group.rvs(d) for d in d_tuple])
r_tuple = tuple([np.random.rand(d) for d in d_tuple])
U2_tuple = tuple([unitary_group.rvs(d) for d in d_tuple])
alpha_tuple = tuple([np.random.rand(d) for d in d_tuple])
xi_tuple = tuple([np.random.rand(d) for d in d_tuple])

parameters = list(
    zip(d_tuple, x_tuple, U1_tuple, r_tuple, U2_tuple, alpha_tuple, xi_tuple)
)


@pytest.mark.parametrize("d, x, U1, r, U2, alpha, xi", parameters)
def PIQUASSO_benchmark(benchmark, cutoff, d, x, U1, r, U2, alpha, xi):
    @benchmark
    def func():
        alpha_ = tf.Variable(alpha)

        simulator = pq.TensorflowPureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()
            pq.Q() | pq.Displacement(alpha=x)
            pq.Q() | pq.Interferometer(U1)
            pq.Q() | pq.Squeezing(r=r)
            pq.Q() | pq.Interferometer(U2)
            pq.Q() | pq.Displacement(alpha=alpha_)
            pq.Q() | pq.Kerr(xi=xi)

        with tf.GradientTape() as tape:
            state = simulator.execute(program).state

            mean_photon_number = state.mean_photon_number()

        jacobian = tape.jacobian(mean_photon_number, alpha_)

        assert (
            jacobian[0] is not None
        ), "The differentiation results in 'None' if the graph gets disjoint."


@pytest.mark.parametrize("d, x, U1, r, U2, alpha, xi", parameters)
def STRAWBERRY_FIELDS_benchmark(benchmark, cutoff, d, x, U1, r, U2, alpha, xi):
    @benchmark
    def func():
        new_program = sf.Program(d)

        var_list = []
        program_param_list = []
        mapping = {}

        for mode in range(d):
            var = tf.Variable(alpha[mode])
            var_list.append(var)
            program_param_list.append(new_program.params(f"alpha_{mode}"))
            mapping[f"alpha_{mode}"] = var

        new_engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})

        with new_program.context as q:
            for mode in range(d):
                sf.ops.Coherent(x[mode]) | q[mode]

            sf.ops.Interferometer(U1) | tuple(q[mode] for mode in range(d))

            for mode in range(d):
                sf.ops.Sgate(r[mode]) | q[mode]

            sf.ops.Interferometer(U2) | tuple(q[mode] for mode in range(d))

            for mode in range(d):
                sf.ops.Dgate(program_param_list[mode]) | q[mode]

            for mode in range(d):
                sf.ops.Kgate(xi[mode]) | q[mode]

        with tf.GradientTape() as tape:
            result = new_engine.run(new_program, args=mapping)
            state = result.state
            mean_photon_numbers = []
            for mode in range(d):
                mean_photon_numbers.append(state.mean_photon(mode))

            mean_photon_numbers = tf.stack(mean_photon_numbers)

        jacobian = tape.jacobian(mean_photon_numbers, var_list)

        assert (
            jacobian[0] is not None
        ), "The differentiation results in 'None' if the graph gets disjoint."
