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


pytestmark = pytest.mark.benchmark(
    group="pure-fock-homodyne",
)


@pytest.fixture
def cutoff():
    return 7


@pytest.fixture
def shots():
    return 100


@pytest.mark.parametrize("d", tuple(range(1, 7)))
def homodyne_benchmark(benchmark, cutoff, d, shots, generate_unitary_matrix):
    @benchmark
    def _():
        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            for i in range(d):
                pq.Q(i) | pq.Displacement(r=0.1) | pq.Squeezing(r=0.2)

            pq.Q(all) | pq.Interferometer(generate_unitary_matrix(d))

            for i in range(d):
                pq.Q(i) | pq.Kerr(xi=0.3)

            pq.Q() | pq.HomodyneMeasurement()

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        simulator_fock.execute(program, shots=shots)
