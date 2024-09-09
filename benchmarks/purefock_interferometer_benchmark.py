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
    group="fock-interferometer",
)


@pytest.fixture
def interferometer():
    return np.array(
        [
            [
                -0.11035524 + 0.43053175j,
                0.16672794 - 0.47819775j,
                0.01831264 - 0.07497556j,
                0.44214383 + 0.23308614j,
                0.49732734 - 0.20707815j,
            ],
            [
                -0.2226116 - 0.39735917j,
                0.30063104 - 0.55866103j,
                0.00329231 - 0.28009116j,
                0.23376629 - 0.22854039j,
                -0.43588261 + 0.12139052j,
            ],
            [
                0.24737463 - 0.24638855j,
                -0.31444021 - 0.28467995j,
                -0.16425914 - 0.33655956j,
                -0.26884137 + 0.68435951j,
                -0.07230938 - 0.10989756j,
            ],
            [
                0.34411568 + 0.00861778j,
                -0.26208576 - 0.04842002j,
                0.64971999 - 0.05882925j,
                0.29164646 + 0.0682574j,
                0.01229276 + 0.54314998j,
            ],
            [
                -0.59541546 - 0.01014536j,
                -0.28295784 + 0.10016806j,
                0.57604147 - 0.13381814j,
                -0.07509227 + 0.08557502j,
                -0.12237335 - 0.42143858j,
            ],
        ]
    )


def piquasso_benchmark(benchmark, pq_purefock_simulator, interferometer):
    @benchmark
    def func():
        with pq.Program() as new_program:
            pq.Q() | pq.StateVector(occupation_numbers=(1, 1, 1, 0, 0))
            pq.Q() | pq.Interferometer(interferometer)

        pq_purefock_simulator.execute(new_program)


def strawberryfields_benchmark(benchmark, d, cutoff, interferometer, sf):
    @benchmark
    def func():
        new_program = sf.Program(d)
        new_engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

        with new_program.context as q:
            sf.ops.Fock(1) | q[0]
            sf.ops.Fock(1) | q[1]
            sf.ops.Fock(1) | q[2]
            sf.ops.Fock(0) | q[3]
            sf.ops.Fock(0) | q[4]

            sf.ops.Interferometer(interferometer) | (q[0], q[1], q[2], q[3], q[4])

        new_engine.run(new_program)
