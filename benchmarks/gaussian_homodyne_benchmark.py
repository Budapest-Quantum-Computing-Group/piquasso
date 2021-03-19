#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import piquasso as pq
import strawberryfields as sf


pytestmark = pytest.mark.benchmark(
    group="gaussian-homodyne-measurement",
)


def piquasso_benchmark(
    benchmark, example_gaussian_pq_program
):
    example_gaussian_pq_program.execute()

    with pq.Program() as new_program:
        pq.Q() | example_gaussian_pq_program.state

        # TODO: Support rotation by an angle, too.
        pq.Q(0) | pq.MeasureHomodyne()

    results = benchmark(new_program.execute)

    assert results


def strawberryfields_benchmark(
    benchmark, example_gaussian_sf_program_and_engine, d
):
    program, engine = example_gaussian_sf_program_and_engine

    results = engine.run(program)

    new_program = sf.Program(d)

    new_program.state = results.state

    with new_program.context as q:
        sf.ops.MeasureHomodyne(phi=0) | q[0]

    benchmark(engine.run, new_program)
