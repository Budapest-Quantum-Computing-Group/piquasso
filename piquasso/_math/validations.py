#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from typing import Iterable, Union, Tuple, Sequence

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

    original_occupation_numbers = tuple(int(number) for number in occupation_numbers)
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
            f"'{cutoff}'. Consider increasing the cutoff via "
            f"`pq.Config(cutoff={required_cutoff})` when creating the simulator."
        )
        if context:
            message += context
        raise InvalidState(message)


def validate_postselection_cutoff(
    cutoff: int,
    photon_counts: Sequence[int],
    occupation_numbers: Iterable[Sequence[int]],
    *,
    context: str = "",
) -> None:
    """Validate that postselection leaves enough Fock-space truncation.

    After post-selecting ``photon_counts``, the effective cutoff becomes
    ``cutoff - sum(photon_counts)``. Each prepared occupation number must
    have fewer photons on the remaining modes than this effective cutoff.

    Args:
        cutoff: The current Fock-space cutoff before postselection.
        photon_counts: The photon numbers being postselected.
        occupation_numbers: Prepared occupation numbers on all modes.
        context: Optional message appended to raised errors.

    Raises:
        InvalidState: If postselection would leave an invalid cutoff or if any
            prepared state would exceed the truncated Fock space afterwards.
    """
    postselected_photons = int(np.sum(photon_counts))
    remaining_cutoff = cutoff - postselected_photons

    if remaining_cutoff <= 0:
        required_cutoff = postselected_photons + 1
        message = (
            f"Post-selecting {postselected_photons} photon(s) on "
            f"{photon_counts} requires a cutoff of at least "
            f"'{required_cutoff}', but the provided cutoff is '{cutoff}'. "
            f"Consider increasing the cutoff via "
            f"`pq.Config(cutoff={required_cutoff})` when creating the simulator."
        )
        if context:
            message += context
        raise InvalidState(message)

    for original_occupation_numbers in occupation_numbers:
        occupation_numbers_tuple = tuple(
            int(number) for number in original_occupation_numbers
        )
        total_photons = int(np.sum(occupation_numbers_tuple))
        remaining_photons = total_photons - postselected_photons

        if remaining_photons >= remaining_cutoff:
            required_cutoff = total_photons + 1
            message = (
                f"After post-selecting {postselected_photons} photon(s) on "
                f"{photon_counts}, the remaining occupation numbers "
                f"'{occupation_numbers_tuple}' require a cutoff of at least "
                f"'{required_cutoff}', but the provided cutoff is '{cutoff}'. "
                f"Consider increasing the cutoff via "
                f"`pq.Config(cutoff={required_cutoff})` when creating the simulator."
            )
            if context:
                message += context
            raise InvalidState(message)
