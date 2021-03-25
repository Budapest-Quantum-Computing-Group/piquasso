#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import piquasso as pq
import strawberryfields as sf


pytestmark = pytest.mark.benchmark(
    group="gaussian-boson-sampling",
)


def piquasso_benchmark(
    benchmark, example_gaussian_pq_program
):
    example_gaussian_pq_program.execute()

    with pq.Program() as new_program:
        pq.Q() | example_gaussian_pq_program.state

        # NOTE: cutoff=5 in PQ corresponds to cutoff=6 in SF.
        # Moreover, in SF we couldn't specify the cutoff, unfortunately.
        pq.Q(0, 1, 2) | pq.MeasureParticleNumber(cutoff=5, shots=4)

    benchmark(new_program.execute)


def strawberryfields_benchmark(
    benchmark, example_gaussian_sf_program_and_engine, d
):
    program, engine = example_gaussian_sf_program_and_engine

    results = engine.run(program)

    new_program = sf.Program(d)

    new_program.state = results.state

    with new_program.context as q:
        sf.ops.MeasureFock() | (q[0], q[1], q[2])

    # NOTE: With 5 modes, SF couldn't support more than 4 shots, due to a bug (?)
    results = benchmark(engine.run, new_program, shots=4)
