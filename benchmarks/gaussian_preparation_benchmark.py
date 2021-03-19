#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest


pytestmark = pytest.mark.benchmark(
    group="gaussian-preparation",
)


def simple_piquasso_benchmark(benchmark, example_gaussian_pq_program):
    benchmark(example_gaussian_pq_program.execute)


def simple_strawberryfields_benchmark(
    benchmark, example_gaussian_sf_program_and_engine
):
    program, engine = example_gaussian_sf_program_and_engine

    benchmark(engine.run, program)
