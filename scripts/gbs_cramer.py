
#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import rpy2.robjects.numpy2ri
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

import piquasso as pq
import strawberryfields as sf

rpy2.robjects.numpy2ri.activate()


d = 5
shots = 1000


with pq.Program() as pq_program:
    pq.Q() | pq.GaussianState(d=d)

    pq.Q(0) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)
    pq.Q(1) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)
    pq.Q(2) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)
    pq.Q(3) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)
    pq.Q(4) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)

    # NOTE: we need to tweak the parameters here a bit, because we use a different
    # definition for the beamsplitter.
    pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, np.pi - 0.06786053071484363)
    pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, np.pi - 1.453770233324797)
    pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, np.pi - 1.2863559998816205)
    pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, np.pi - 0.5236836466492961)
    pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715,  np.pi - 4.481575657714487)
    pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, np.pi - 1.5073556513699888)
    pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, np.pi - 1.9550229282085838)
    pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504,  np.pi - 3.289367083610399)

    # NOTE: In SF the cutoff is 5, and couldn't be changed.
    pq.Q(0, 1, 2) | pq.MeasureParticleNumber(cutoff=5, shots=shots)


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


def to_r_matrix(A):
    nr, nc = A.shape
    rA = ro.r.matrix(A, nrow=nr, ncol=nc)
    ro.r.assign("x", rA)
    return rA


def cramer_multidim_test(x, y):
    # NOTE: Uncomment below to install 'cramer'.
    # utils = importr('utils')
    # utils.install_packages('cramer')
    importr('cramer')

    xR = to_r_matrix(x)
    yR = to_r_matrix(y)

    cramer_test = ro.r['cramer.test']
    result = cramer_test(xR, yR)
    print(result)


if __name__ == "__main__":

    pq_results = np.array(pq_program.execute()[0].outcome)
    sf_results = sf_engine.run(sf_program, shots=shots).samples

    cramer_multidim_test(pq_results, sf_results)
