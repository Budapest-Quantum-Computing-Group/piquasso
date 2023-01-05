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


pytestmark = pytest.mark.benchmark(
    group="fock-beamsplitter",
)


def piquasso_benchmark(benchmark, pq_purefock_simulator):
    @benchmark
    def func():
        with pq.Program() as new_program:
            pq.Q() | pq.StateVector(occupation_numbers=(1, 1, 1, 0, 0))

            pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
            pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
            pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
            pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
            pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
            pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
            pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
            pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        pq_purefock_simulator.execute(new_program)


def strawberryfields_benchmark(benchmark, d, cutoff):
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

            sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
            sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
            sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
            sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
            sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
            sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
            sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
            sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

        new_engine.run(new_program)
