#
# Copyright 2021 Budapest Quantum Computing Group
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

import numpy as np

import piquasso as pq
import strawberryfields as sf


def gbs_hypothesis_test_script(cramer_hypothesis_test):
    d = 5
    shots = 1000

    pq_state = pq.GaussianState(d=d)

    with pq.Program() as pq_program:
        pq.Q(all) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715,  4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504,  3.289367083610399)

        # NOTE: In SF the cutoff is 5, and couldn't be changed.
        pq.Q(0, 1, 2) | pq.ParticleNumberMeasurement(cutoff=5)

    sf_program = sf.Program(d)
    sf_engine = sf.Engine(backend="gaussian")

    with sf_program.context as q:
        sf.ops.Sgate(0.1) | q[0]
        sf.ops.Sgate(0.1) | q[1]
        sf.ops.Sgate(0.1) | q[2]
        sf.ops.Sgate(0.1) | q[3]
        sf.ops.Sgate(0.1) | q[4]

        sf.ops.Dgate(1) | q[0]
        sf.ops.Dgate(1) | q[1]
        sf.ops.Dgate(1) | q[2]
        sf.ops.Dgate(1) | q[3]
        sf.ops.Dgate(1) | q[4]

        sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
        sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
        sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
        sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
        sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
        sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
        sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
        sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

        sf.ops.MeasureFock() | (q[0], q[1], q[2])

    pq_results = np.array(pq_state.apply(pq_program, shots=shots).samples)
    sf_results = sf_engine.run(sf_program, shots=shots).samples

    accepted = cramer_hypothesis_test(pq_results, sf_results)

    assert accepted, "The hypothesis is not accepted."
