#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import strawberryfields as sf
import piquasso as pq


@pytest.fixture
def d():
    return 5


@pytest.fixture
def example_gaussian_pq_program(d):
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=d)

        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        # NOTE: we need to tweak the parameters here a bit, because we use a different
        # definition for the beamsplitter.
        pq.Q(0, 1) | pq.B(0.0959408065906761, np.pi - 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, np.pi - 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, np.pi - 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, np.pi - 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715,  np.pi - 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, np.pi - 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, np.pi - 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504,  np.pi - 3.289367083610399)

    yield program

    # TODO: The state has to be reset, because the setup runs only once at the beginning
    # of the calculations, therefore the same `GaussianState` instance will be used.
    program.state.reset()


@pytest.fixture
def example_gaussian_sf_program_and_engine(d):
    """
    NOTE: the covariance matrix SF is returning is half of ours...
    It seems that our implementation is OK, however.
    """

    program = sf.Program(d)
    engine = sf.Engine(backend="gaussian")

    with program.context as q:
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

    return program, engine
