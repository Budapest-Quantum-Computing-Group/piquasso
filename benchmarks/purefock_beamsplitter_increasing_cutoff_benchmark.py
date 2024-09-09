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

import numpy as np

import piquasso as pq


pytestmark = pytest.mark.benchmark(
    group="pure-fock",
)


@pytest.fixture
def theta():
    return np.pi / 5


@pytest.fixture
def d():
    return 5


@pytest.mark.parametrize("cutoff", range(3, 14))
def piquasso_benchmark(benchmark, d, cutoff, theta):
    @benchmark
    def func():
        state_vector = [cutoff // d] * d
        state_vector[0] += cutoff % d - 1

        with pq.Program() as program:
            pq.Q(all) | pq.StateVector(state_vector)
            for i in range(d - 1):
                pq.Q(i, i + 1) | pq.Beamsplitter(theta)

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        simulator_fock.execute(program)


@pytest.mark.parametrize("cutoff", (3, 4, 5))
def strawberryfields_benchmark(benchmark, d, cutoff, theta, sf):
    @benchmark
    def func():
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

        circuit = sf.Program(d)

        state_vector = [cutoff // d] * d
        state_vector[0] += cutoff % d - 1

        with circuit.context as q:
            for i, n in enumerate(state_vector):
                sf.ops.Fock(n) | q[i]

            for w in range(d - 1):
                sf.ops.BSgate(theta) | (q[w], q[w + 1])

        eng.run(circuit).state
