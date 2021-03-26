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
    benchmark, example_pq_gaussian_state
):
    @benchmark
    def func():
        with pq.Program() as new_program:
            pq.Q() | example_pq_gaussian_state

            # TODO: Support rotation by an angle, too.
            pq.Q(0) | pq.MeasureHomodyne()

            new_program.execute()


def strawberryfields_benchmark(
    benchmark, example_sf_gaussian_state, d
):
    @benchmark
    def func():
        new_program = sf.Program(d)
        new_engine = sf.Engine(backend="gaussian")

        new_program.state = example_sf_gaussian_state

        with new_program.context as q:
            sf.ops.MeasureHomodyne(phi=0) | q[0]

        new_engine.run(new_program)
