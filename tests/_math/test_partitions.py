#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

from piquasso._math.combinatorics import partitions, partitions_bounded_k


def test_partitions():
    boxes = 3
    particles = 4

    assert np.all(
        partitions(boxes, particles)
        == [
            (4, 0, 0),
            (3, 1, 0),
            (3, 0, 1),
            (2, 2, 0),
            (2, 1, 1),
            (2, 0, 2),
            (1, 3, 0),
            (1, 2, 1),
            (1, 1, 2),
            (1, 0, 3),
            (0, 4, 0),
            (0, 3, 1),
            (0, 2, 2),
            (0, 1, 3),
            (0, 0, 4),
        ]
    )


def test_partitions_bounded_k():
    boxes = 4
    particles = 5
    constrained_boxes = (2, 3)
    max_per_box = (1, 1)

    k_limit = 2

    assert np.all(
        partitions_bounded_k(boxes, particles, constrained_boxes, max_per_box, k_limit)
        == [
            [5, 0, 0, 0],
            [4, 1, 0, 0],
            [4, 0, 1, 0],
            [4, 0, 0, 1],
            [3, 2, 0, 0],
            [3, 1, 1, 0],
            [3, 1, 0, 1],
            [3, 0, 1, 1],
            [2, 3, 0, 0],
            [2, 2, 1, 0],
            [2, 2, 0, 1],
            [2, 1, 1, 1],
            [1, 4, 0, 0],
            [1, 3, 1, 0],
            [1, 3, 0, 1],
            [1, 2, 1, 1],
            [0, 5, 0, 0],
            [0, 4, 1, 0],
            [0, 4, 0, 1],
            [0, 3, 1, 1],
        ]
    )
