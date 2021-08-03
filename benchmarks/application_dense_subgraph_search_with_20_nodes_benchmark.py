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
import networkx as nx
import piquasso as pq
from strawberryfields.apps import sample, subgraph

pytestmark = pytest.mark.benchmark(
    group="application_dense_subgraph_search_with_20_nodes",
)

def piquasso_benchmark(benchmark, adj_matrix_20):
    @benchmark
    def func():
        pq.constants.seed(40)
        adjacency_matrix = adj_matrix_20

        with pq.Program() as program:
            pq.Q() | pq.Graph(adjacency_matrix)
            pq.Q() | pq.ParticleNumberMeasurement()

        state = pq.GaussianState(d=len(adjacency_matrix))
        result = state.apply(program, shots=10)
        state.validate()

        # Convert sample results to subgraph nodes
        subgraphs = result.to_subgraph_nodes()

        # Create nx graph from the adjacency_matrix
        TA_graph = nx.Graph(adjacency_matrix)

        # Search for dense subgraphs with size between 2 and 3
        result.dense_search(subgraphs, TA_graph, 10, 12, max_count=3)


def strawberryfields_benchmark(benchmark, adj_matrix_20):
    @benchmark
    def func():
        adjacency_matrix = adj_matrix_20
        TA_graph = nx.Graph(adjacency_matrix)
        samples = sample.sample(adjacency_matrix, 10, 10)
        subgraphs = sample.to_subgraphs(samples, TA_graph)
        subgraph.search(subgraphs, TA_graph, 10, 12, max_count=3)
