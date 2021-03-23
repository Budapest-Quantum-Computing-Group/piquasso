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

        # TODO: With cutoff=6, it is embarassingly slow :/
        # Also, with SF, we couldn't specify the cutoff, unfortunately
        pq.Q(0, 1, 2) | pq.MeasureParticleNumber(cutoff=3)

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

    benchmark(engine.run, new_program)
