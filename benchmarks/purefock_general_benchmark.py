#
# Copyright 2021-2023 Budapest Quantum Computing Group
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
import strawberryfields as sf

from scipy.stats import unitary_group


pytestmark = pytest.mark.benchmark(
    group="pure-fock-general",
)


@pytest.fixture
def alpha():
    return 0.1


@pytest.fixture
def r():
    return 0.2


@pytest.fixture
def xi():
    return 0.3


parameters = [(d, unitary_group.rvs(d)) for d in range(3, 7)]


@pytest.mark.parametrize("d, interferometer", parameters)
def piquasso_benchmark(benchmark, d, interferometer, r, alpha, xi):
    @benchmark
    def func():
        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            for i in range(d):
                pq.Q(i) | pq.Displacement(r=alpha) | pq.Squeezing(r)

            pq.Q(all) | pq.Interferometer(interferometer)

            for i in range(d):
                pq.Q(i) | pq.Kerr(xi)

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=d))

        simulator_fock.execute(program)


@pytest.mark.parametrize("d, interferometer", parameters)
def strawberryfields_benchmark(benchmark, d, interferometer, r, alpha, xi):
    @benchmark
    def func():
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": d})

        circuit = sf.Program(d)

        with circuit.context as q:
            for i in range(d):
                sf.ops.Dgate(alpha) | q[i]
                sf.ops.Sgate(r) | q[i]

            sf.ops.Interferometer(interferometer) | tuple(q[i] for i in range(d))

            for i in range(d):
                sf.ops.Kgate(xi) | q[i]

        eng.run(circuit)
