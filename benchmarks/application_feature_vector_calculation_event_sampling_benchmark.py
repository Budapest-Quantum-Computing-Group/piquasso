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
from strawberryfields.apps import sample, similarity

pytestmark = pytest.mark.benchmark(
    group="application_feature_vector_calculation_event_sampling",
)

def piquasso_benchmark(
        benchmark, adjacency_matrix,
):
    @benchmark
    def func():
        pq.constants.seed(40)

        with pq.Program() as program:
            pq.Q() | pq.Graph(adjacency_matrix)
            pq.Q() | pq.ParticleNumberMeasurement()

        state = pq.GaussianState(d=len(adjacency_matrix))
        result = state.apply(program)
        state.validate()

        result.feature_vector_events_sampling([2, 4], 2)


def strawberryfields_benchmark(
        benchmark, adjacency_matrix
):
    @benchmark
    def func():
        samples = sample.sample(adjacency_matrix, 5, 4)
        similarity.feature_vector_events_sampling(samples, [2, 4], 2)
