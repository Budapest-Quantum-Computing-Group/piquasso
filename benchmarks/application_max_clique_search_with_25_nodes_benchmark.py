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
import numpy as np
import networkx as nx
import piquasso as pq
from strawberryfields.apps import sample, clique

pytestmark = pytest.mark.benchmark(
    group="application_maximum_clique_search_with_25_nodes",
)

def piquasso_benchmark(benchmark, adj_matrix_25):
    @benchmark
    def func():
        pq.constants.seed(40)
        adjacency_matrix = adj_matrix_25

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

        # Shrink subgraphs until they are cliques
        shrunk = [result.shrink(s, TA_graph) for s in subgraphs]

        # Iteratively search for bigger cliques
        [result.search(s, TA_graph, 10) for s in shrunk]


def strawberryfields_benchmark(benchmark, adj_matrix_25):
    @benchmark
    def func():
        adjacency_matrix = adj_matrix_25
        TA_graph = nx.Graph(adjacency_matrix)
        samples = sample.sample(adjacency_matrix, 25, 10)
        subgraphs = sample.to_subgraphs(samples, TA_graph)
        shrunk = [clique.shrink(s, TA_graph) for s in subgraphs]
        [clique.search(s, TA_graph, 10) for s in shrunk]
