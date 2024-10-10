#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from typing import Iterable, Union

import numpy as np


def is_natural(number: Union[int, float]) -> bool:
    return bool(np.isclose(number % 1, 0.0) and round(number) >= 0)


def all_natural(array: Iterable) -> bool:
    return all(is_natural(number) for number in array)


def is_zero_or_one(number: Union[int, float]) -> bool:
    return bool(np.isclose(number, 0.0) or np.isclose(number, 1.0))


def all_zero_or_one(array: Iterable) -> bool:
    return all(is_zero_or_one(number) for number in array)


def all_real_and_positive(vector: Iterable) -> bool:
    return all(element >= 0.0 or np.isclose(element, 0.0) for element in vector)


def all_in_interval(vector: Iterable, lower: float, upper: float) -> bool:
    return all(
        (element >= lower or np.isclose(element, lower))
        and (element <= upper or np.isclose(element, upper))
        for element in vector
    )
