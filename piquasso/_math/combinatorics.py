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

from itertools import chain, combinations, combinations_with_replacement
from typing import Tuple, Iterable, Iterator, TypeVar, List, Type

import numpy as np

_T = TypeVar("_T")


def powerset(iterable: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(iterable, r) for r in range(len(s) + 1))


def partitions(
    boxes: int, particles: int, class_: Type[tuple] = tuple
) -> List[Tuple[int, ...]]:
    if particles == 0:
        return [class_([0] * boxes)]

    masks = np.rot90(np.identity(boxes, dtype=int))

    return sorted(
        class_(sum(c))  # type: ignore
        for c in combinations_with_replacement(masks, particles)
    )


def get_occupation_numbers(d: int, cutoff: int) -> List[Tuple[int, ...]]:
    return [
        occupation_number
        for particle_number in range(cutoff)
        for occupation_number in partitions(d, particle_number)
    ]
