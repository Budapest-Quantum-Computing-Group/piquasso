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

import numpy as np
import piquasso as pq


pytestmark = pytest.mark.benchmark(
    group="gaussian-homodyne-measurement",
)


def piquasso_benchmark(benchmark, pq_gaussian_simulator, example_pq_gaussian_state):
    @benchmark
    def func():
        with pq.Program() as new_program:
            pq.Q(0) | pq.HomodyneMeasurement(phi=np.pi / 4)

            pq_gaussian_simulator.execute(
                new_program, initial_state=example_pq_gaussian_state
            )


def strawberryfields_benchmark(benchmark, example_sf_gaussian_state, d, sf):
    @benchmark
    def func():
        new_program = sf.Program(d)
        new_engine = sf.Engine(backend="gaussian")

        new_program.state = example_sf_gaussian_state

        with new_program.context as q:
            sf.ops.MeasureHomodyne(phi=np.pi / 4) | q[0]

        new_engine.run(new_program)
