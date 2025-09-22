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

from typing import Iterable, Union, Tuple

import numpy as np

from piquasso.api.exceptions import InvalidState


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


def are_modes_consecutive(modes: Tuple[int, ...]) -> bool:
    expected = np.arange(modes[0], modes[-1] + 1)

    return len(modes) == len(expected) and bool(np.all(modes == expected))


def validate_occupation_numbers(
    occupation_numbers: Iterable, d: int, cutoff: int, context: str = ""
) -> None:
    """Validate occupation numbers for Fock state preparation.

    Args:
        occupation_numbers: Sequence of occupation numbers to validate.
        d: The expected number of modes.
        cutoff: The cutoff dimension for the Fock space.
        context: Optional message appended to raised errors.

    Raises:
        InvalidState: If the length of ``occupation_numbers`` does not match
            ``d`` or if the total particle number requires a larger cutoff.
    """

    original_occupation_numbers = tuple(occupation_numbers)
    occupation_numbers = np.array(original_occupation_numbers)

    if len(occupation_numbers) != d:
        message = (
            f"The occupation numbers '{original_occupation_numbers}' are "
            f"not well-defined on '{d}' modes."
        )
        if context:
            message += context
        raise InvalidState(message)

    total = int(np.sum(occupation_numbers))
    if total >= cutoff:
        required_cutoff = total + 1
        message = (
            f"The occupation numbers '{original_occupation_numbers}' require "
            f"a cutoff of at least '{required_cutoff}', but the provided cutoff is "
            f"'{cutoff}'."
        )
        if context:
            message += context
        raise InvalidState(message)
