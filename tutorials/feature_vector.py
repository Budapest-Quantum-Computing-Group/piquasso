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

import numpy as np
import piquasso as pq
import networkx as nx

adjacency_matrix = np.array(
    [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]
)
shots = 15

pq.constants.seed(40)

with pq.Program() as program:
    pq.Q() | pq.Graph(adjacency_matrix)
    pq.Q() | pq.ParticleNumberMeasurement()

state = pq.GaussianState(d=4)
result = state.apply(program)
state.validate()

print(result.samples)
print(result.feature_vector_events_sampling([2, 4], 2))
print(result.feature_vector_orbits_sampling([[1, 1], [2], [1, 1, 1, 1], [2, 1, 1]]))
print(result.feature_vector_events(nx.Graph(adjacency_matrix), [2, 4], 2))
print(result.feature_vector_events(nx.Graph(adjacency_matrix), [2, 4], 2, samples=1000))
