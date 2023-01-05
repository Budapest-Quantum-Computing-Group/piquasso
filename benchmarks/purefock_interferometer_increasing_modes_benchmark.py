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

import numpy as np

import piquasso as pq
import strawberryfields as sf

from scipy.stats import unitary_group


pytestmark = pytest.mark.benchmark(
    group="pure-fock-interferometer-increasing-modes",
)


@pytest.fixture
def theta():
    return np.pi / 5


@pytest.fixture
def cutoff():
    return 4


parameters = [(d, unitary_group.rvs(d)) for d in range(3, 6)]


@pytest.mark.parametrize("d, interferometer", parameters)
def piquasso_benchmark(benchmark, d, interferometer, cutoff, theta):
    @benchmark
    def func():
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector([1] * 3 + [0] * (d - 3))

            pq.Q(all) | pq.Interferometer(interferometer)

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        simulator_fock.execute(program)


@pytest.mark.parametrize("d, interferometer", parameters)
def strawberryfields_benchmark(benchmark, d, interferometer, cutoff, theta):
    @benchmark
    def func():
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

        circuit = sf.Program(d)

        with circuit.context as q:
            sf.ops.Fock(1) | q[0]
            sf.ops.Fock(1) | q[1]
            sf.ops.Fock(1) | q[2]

            sf.ops.Interferometer(interferometer) | tuple(q[i] for i in range(d))

        eng.run(circuit)
