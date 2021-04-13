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

from itertools import chain, combinations, combinations_with_replacement


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def partitions(boxes, particles, class_=tuple):
    if particles == 0:
        return [class_([0] * boxes)]

    masks = np.rot90(np.identity(boxes, dtype=int))

    return sorted(
        class_(sum(c)) for c in combinations_with_replacement(masks, particles)
    )
