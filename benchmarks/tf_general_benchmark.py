#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

import piquasso as pq
import tensorflow as tf

from scipy.stats import unitary_group


pytestmark = pytest.mark.benchmark(
    group="tf-general",
)


@pytest.fixture
def alpha():
    return 0.01


@pytest.fixture
def r():
    return 0.01


@pytest.fixture
def xi():
    return 0.3


parameters = [(d, unitary_group.rvs(d)) for d in range(3, 5)]


@pytest.mark.parametrize("d, interferometer", parameters)
def piquasso_benchmark(benchmark, d, interferometer, alpha, r, xi):
    @benchmark
    def func():
        alpha_ = tf.Variable(alpha)

        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            for i in range(d):
                pq.Q(i) | pq.Displacement(r=alpha) | pq.Squeezing(r)

            pq.Q(all) | pq.Interferometer(interferometer)

            for i in range(d):
                pq.Q(i) | pq.Kerr(xi)

        simulator_fock = pq.PureFockSimulator(
            d=d, config=pq.Config(cutoff=d), connector=pq.TensorflowConnector()
        )

        with tf.GradientTape() as tape:
            state = simulator_fock.execute(program).state
            mean_photon_number = state.mean_photon_number()

        tape.gradient(mean_photon_number, [alpha_])


@pytest.mark.parametrize("d, interferometer", parameters)
def strawberryfields_benchmark(benchmark, d, interferometer, alpha, r, xi, sf):
    @benchmark
    def func():
        program = sf.Program(d)

        mapping = {}

        alpha_ = tf.Variable(alpha)
        param = program.params("alpha")
        mapping["alpha"] = alpha_

        engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": d})

        with program.context as q:
            for i in range(d):
                sf.ops.Dgate(param) | q[i]
                sf.ops.Sgate(r) | q[i]

            sf.ops.Interferometer(interferometer) | tuple(q[i] for i in range(d))

            for i in range(d):
                sf.ops.Kgate(xi) | q[i]

        with tf.GradientTape() as tape:
            result = engine.run(program, args=mapping)
            state = result.state
            mean = sum([state.mean_photon(mode)[0] for mode in range(d)])

        tape.gradient(mean, [alpha_])
