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

from typing import Iterable

import itertools


def is_ordered_sublist(smaller_list: Iterable, larger_list: Iterable) -> bool:
    left_intersection = [element for element in smaller_list if element in larger_list]
    right_intersection = [element for element in larger_list if element in smaller_list]

    return left_intersection == right_intersection


def deduplicate_neighbours(redundant_list: Iterable) -> Iterable:
    return [group[0] for group in itertools.groupby(redundant_list)]
