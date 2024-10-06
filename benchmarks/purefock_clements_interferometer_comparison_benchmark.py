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
from piquasso.decompositions.clements import clements

from scipy.stats import unitary_group

pytestmark = pytest.mark.benchmark(
    group="pure-fock-clements-comparison",
)


@pytest.fixture
def cutoff():
    return 5


d_tuple = (2, 3, 4, 5)
U_tuple = tuple([unitary_group.rvs(d) for d in d_tuple])


@pytest.mark.parametrize("d, U", zip(d_tuple, U_tuple))
def piquasso_interferometer_benchmark(benchmark, d, cutoff, U):
    @benchmark
    def func():
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            for i in range(d):
                pq.Q(i) | pq.Squeezing(r=0.1)

            pq.Q() | pq.Interferometer(U)

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        simulator_fock.execute(program)


@pytest.mark.parametrize("d, U", zip(d_tuple, U_tuple))
def piquasso_clements_benchmark(benchmark, d, cutoff, U):
    decomposition = clements(U, connector=pq.NumpyConnector())

    @benchmark
    def func():
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            for i in range(d):
                pq.Q(i) | pq.Squeezing(r=0.1)

            for operation in decomposition.first_beamsplitters:
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
                pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

            for operation in decomposition.middle_phaseshifters:
                pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

            for operation in decomposition.last_beamsplitters:
                pq.Q(*operation.modes) | pq.Beamsplitter(-operation.params[0], 0.0)
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=-operation.params[1])

        simulator_fock = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        simulator_fock.execute(program)
