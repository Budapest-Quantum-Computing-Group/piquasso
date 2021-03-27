#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import piquasso as pq
import strawberryfields as sf
import numpy as np


pytestmark = pytest.mark.benchmark(
    group="gaussian-threshold-detection",
)


def piquasso_benchmark(
    benchmark, d
):
    @benchmark
    def func():
        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=d)

            pq.Q(0) | pq.Squeezing(r=1.0, phi=np.pi/1)
            pq.Q(1) | pq.Squeezing(r=0.2, phi=np.pi/2)
            pq.Q(2) | pq.Squeezing(r=0.3, phi=np.pi/3)
            pq.Q(3) | pq.Squeezing(r=0.4, phi=np.pi/4)
            pq.Q(4) | pq.Squeezing(r=1.0, phi=np.pi/5)

            # NOTE: we need to tweak the parameters here a bit, because we use a
            # different definition for the beamsplitter.
            pq.Q(0, 1) | pq.Beamsplitter(
                0.0959408065906761, np.pi - 0.06786053071484363
            )
            pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, np.pi - 1.453770233324797)
            pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, np.pi - 1.2863559998816205)
            pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, np.pi - 0.5236836466492961)
            pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715,  np.pi - 4.481575657714487)
            pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, np.pi - 1.5073556513699888)
            pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, np.pi - 1.9550229282085838)
            pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504,  np.pi - 3.289367083610399)

            pq.Q(0, 1, 2) | pq.MeasureThreshold(shots=20)

        program.execute()


def strawberryfields_benchmark(
    benchmark, d
):
    @benchmark
    def func():
        program = sf.Program(d)
        engine = sf.Engine(backend="gaussian")

        with program.context as q:
            sf.ops.Sgate(1.0, phi=np.pi/1) | q[0]
            sf.ops.Sgate(0.2, phi=np.pi/2) | q[1]
            sf.ops.Sgate(0.3, phi=np.pi/3) | q[2]
            sf.ops.Sgate(0.4, phi=np.pi/4) | q[3]
            sf.ops.Sgate(1.0, phi=np.pi/5) | q[4]

            sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
            sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
            sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
            sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
            sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
            sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
            sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
            sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

            sf.ops.MeasureThreshold() | (q[0], q[1], q[2])

        engine.run(program, shots=20)
