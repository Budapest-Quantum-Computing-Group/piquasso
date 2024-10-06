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


pytestmark = pytest.mark.benchmark(
    group="tensorflow",
)


d_tuple = tuple(range(2, 6))


@pytest.mark.parametrize("d, cutoff", zip(d_tuple, d_tuple))
def piquasso_benchmark(benchmark, cutoff, d):
    @benchmark
    def func():
        r = tf.Variable(0.05)
        simulator = pq.PureFockSimulator(
            d=d, config=pq.Config(cutoff=cutoff), connector=pq.TensorflowConnector()
        )

        with tf.GradientTape() as tape:
            with pq.Program() as program:
                pq.Q() | pq.Vacuum()
                for i in range(d):
                    pq.Q(i) | pq.Displacement(r=r)

            state = simulator.execute(program).state
            mean = state.mean_photon_number()

        tape.gradient(mean, [r])


@pytest.mark.parametrize("d, cutoff", zip(d_tuple, d_tuple))
def strawberryfields_benchmark(benchmark, cutoff, d, sf):
    @benchmark
    def func():
        new_program = sf.Program(d)

        mapping = {}

        alpha = tf.Variable(0.05)
        param = new_program.params("alpha")
        mapping["alpha"] = alpha

        new_engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})

        with new_program.context as q:
            for i in range(d):
                sf.ops.Dgate(param) | q[i]

        with tf.GradientTape() as tape:
            result = new_engine.run(new_program, args=mapping)
            state = result.state
            mean = sum([state.mean_photon(mode)[0] for mode in range(d)])

        tape.gradient(mean, [alpha])
