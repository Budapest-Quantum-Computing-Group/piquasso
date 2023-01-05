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

import subprocess

import numpy as np
import matplotlib.pyplot as plt

import piquasso as pq

import strawberryfields as sf


def threshold_histogram_script():
    d = 5
    shots = 2000

    pq_simulator = pq.GaussianSimulator(d=d)

    with pq.Program() as pq_program:
        pq.Q(0) | pq.Squeezing(r=1.0, phi=np.pi / 1)
        pq.Q(1) | pq.Squeezing(r=0.2, phi=np.pi / 2)
        pq.Q(2) | pq.Squeezing(r=0.3, phi=np.pi / 3)
        pq.Q(3) | pq.Squeezing(r=0.4, phi=np.pi / 4)
        pq.Q(4) | pq.Squeezing(r=1.0, phi=np.pi / 5)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        pq.Q(0, 1, 2) | pq.ThresholdMeasurement()

    sf_program = sf.Program(d)
    sf_engine = sf.Engine(backend="gaussian")

    with sf_program.context as q:
        sf.ops.Sgate(1.0, phi=np.pi / 1) | q[0]
        sf.ops.Sgate(0.2, phi=np.pi / 2) | q[1]
        sf.ops.Sgate(0.3, phi=np.pi / 3) | q[2]
        sf.ops.Sgate(0.4, phi=np.pi / 4) | q[3]
        sf.ops.Sgate(1.0, phi=np.pi / 5) | q[4]

        sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
        sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
        sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
        sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
        sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
        sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
        sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
        sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

        sf.ops.MeasureThreshold() | (q[0], q[1], q[2])

    pq_results = np.array(pq_simulator.execute(pq_program, shots=shots).samples)
    sf_results = sf_engine.run(sf_program, shots=shots).samples

    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(pq_results, bins=n_bins)
    axs[1].hist(sf_results, bins=n_bins)

    fig.savefig("histogram.png")

    subprocess.call(("xdg-open", "histogram.png"))
