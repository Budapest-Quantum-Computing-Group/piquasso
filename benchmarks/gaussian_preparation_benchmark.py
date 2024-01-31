#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

import pytest


pytestmark = pytest.mark.benchmark(
    group="gaussian-preparation",
)


def simple_piquasso_benchmark(
    benchmark, pq_gaussian_simulator, example_gaussian_pq_program
):
    @benchmark
    def _():
        pq_gaussian_simulator.execute(example_gaussian_pq_program)


def simple_strawberryfields_benchmark(
    benchmark, example_gaussian_sf_program_and_engine
):
    program, engine = example_gaussian_sf_program_and_engine

    benchmark(engine.run, program)
