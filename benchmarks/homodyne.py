#!/usr/bin/env python
"""Benchmark and profiling for computing homodyne expectation values on vacuum.

Both frameworks are initialized with the same circuits, and the same outputs are
calculated, which is the quadrature mean and the covariance after a rotation with angle
$`\pi/3`$ on the second mode.

To run this script:

.. code-block:: bash

    ./benchmarks/homodyne.py [NUMBER_OF_ITERATIONS]

"""

import timeit

import numpy as np
import strawberryfields as sf

import sys
import os

sys.path.append(os.getcwd())

import piquasso as pq  # noqa: E402


NO_OF_MODES = 5


def piquasso_setup():
    program = pq.Program(
        state=pq.GaussianState.create_vacuum(d=NO_OF_MODES),
    )

    with program:
        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        pq.Q(0, 1) | pq.B(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504, 3.289367083610399)

    return program


def piquasso_benchmark(program):
    program.execute()

    result = program.state.reduced_rotated_mean_and_cov(
        modes=(1,), phi=np.pi/3
    )

    # TODO: The state has to be reset, because the setup runs only once at the beginning
    # of the calculations, therefore the same `GaussianState` instance will be used.
    program.state.reset()

    return result


def strawberryfields_setup():
    program = sf.Program(NO_OF_MODES)
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


def strawberryfields_benchmark(program, engine):
    results = engine.run(program)

    return results.state.quad_expectation(mode=1, phi=np.pi/3)


def main(iterations):
    pq_time = timeit.timeit(
        "homodyne.piquasso_benchmark(program)",
        setup=(
            "import homodyne;"
            "program = homodyne.piquasso_setup()"
        ),
        number=iterations,
    )
    print("Piquasso (PQ): {}".format(pq_time))

    sf_time = timeit.timeit(
        "homodyne.strawberryfields_benchmark(program, engine)",
        setup=(
            "import homodyne;"
            "program, engine = homodyne.strawberryfields_setup()"
        ),
        number=iterations,
    )
    print("Strawberry Fields (SF): {}".format(sf_time))

    print()

    print("Ratio (SF/PQ): {}".format(sf_time / pq_time))


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    main(iterations)
