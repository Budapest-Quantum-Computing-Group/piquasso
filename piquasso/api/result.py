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


class Result:
    def __init__(self, instruction, samples):
        self.instruction = instruction
        self.samples = samples

    def __repr__(self):
        return f"<Result instruction={self.instruction} samples={self.samples}>"

    def to_subgraph_nodes(self):
        """Convert samples to subgraph modes.

        Assuming that a graph's adjacency matrix is embedded into the circuit, the
        samples from particle number measurement naturally favor dense subgraphs. The
        samples could be converted to subgraph modes using this method.

        The resulting subgraph node indices correspond to the indices in the adjacency
        matrix.

        Only meaningful for discrete samples generated by
        :class:`ParticleNumberMeasurement`.

        References:
            - https://arxiv.org/abs/1803.10731
            - https://arxiv.org/abs/1803.10730

        Returns:
            list: The list of subgraph node indices.
        """
        subgraphs = []

        for sample in self.samples:
            modes = []
            for index, count in enumerate(sample):
                modes += [index] * count

            subgraphs.append(sorted(modes))

        return subgraphs
