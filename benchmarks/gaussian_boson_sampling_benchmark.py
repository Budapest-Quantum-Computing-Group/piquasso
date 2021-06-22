#
# Copyright 2021 Budapest Quantum Computing Group
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

import piquasso as pq
import strawberryfields as sf


pytestmark = pytest.mark.benchmark(
    group="gaussian-boson-sampling",
)


def piquasso_benchmark(
    benchmark, example_pq_gaussian_state
):
    @benchmark
    def func():
        with pq.Program() as new_program:
            pq.Q() | example_pq_gaussian_state

            # NOTE: In SF the cutoff is 5, and couldn't be changed
            pq.Q(0, 1, 2) | pq.ParticleNumberMeasurement(cutoff=5)

        new_program.execute(shots=4)


def strawberryfields_benchmark(
    benchmark, example_sf_gaussian_state, d
):
    @benchmark
    def func():
        new_program = sf.Program(d)
        new_engine = sf.Engine(backend="gaussian")

        new_program.state = example_sf_gaussian_state

        with new_program.context as q:
            sf.ops.MeasureFock() | (q[0], q[1], q[2])

        new_engine.run(new_program, shots=4)
