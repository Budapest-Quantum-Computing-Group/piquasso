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

import numpy as np

from functools import lru_cache

from scipy.special import factorial


def create_single_mode_displacement_matrix(
    r: float,
    phi: float,
    cutoff: int,
) -> np.ndarray:
    r"""
    This method generates the Displacement operator following a recursion rule.
    Reference: https://quantum-journal.org/papers/q-2020-11-30-366/.
    Args:
    r (float): This is the Displacement amplitude. Typically this value can be
        negative or positive depending on the desired displacement direction.
        Note:
            Setting :math:`|r|` to higher values will require you to have a higer
            cuttof dimensions.
    phi (float): This is the Dispalacement angle. Its ranges are
        :math:`\phi \in [ 0, 2 \pi )`
    Returns:
        np.ndarray: The constructed Displacement matrix representing the Fock
        operator.
    """
    cutoff_range = np.arange(cutoff)
    sqrt_indices = np.sqrt(cutoff_range)
    denominator = 1 / np.sqrt(factorial(cutoff_range))

    displacement = r * np.exp(1j * phi)
    displacement_conj = np.conj(displacement)
    columns = []

    columns.append(np.power(displacement, cutoff_range) * denominator)

    roll_index = np.arange(-1, cutoff - 1)

    for _ in range(cutoff - 1):
        columns.append(
            sqrt_indices * columns[-1][roll_index] - displacement_conj * columns[-1]
        )

    return np.exp(-0.5 * r**2) * np.column_stack(columns) * denominator


@lru_cache
def _double_factorial_array(cutoff):
    """
    NOTE: For some reason, this rudimentary implementation is faster than using
    `scipy.special.factorial2`.
    """
    array = np.empty(shape=(cutoff + 1) // 2)

    array[0] = 1

    for i in range(1, len(array)):
        array[i] = (2 * i - 1) / (2 * i) * array[i - 1]

    return array


def create_single_mode_squeezing_matrix(
    r: float, phi: float, cutoff: int, complex_dtype: np.dtype
) -> np.ndarray:
    """
    This method generates the Squeezing operator following a recursion rule.
    Reference: https://quantum-journal.org/papers/q-2020-11-30-366/.

    Args:
    r (float): This is the Squeezing amplitude. Typically this value can be
        negative or positive depending on the desired squeezing direction.
        Note:
            Setting :math:`|r|` to higher values will require you to have a higer
            cuttof dimensions.
    phi (float): This is the Squeezing angle. Its ranges are
        :math:`\phi \in [ 0, 2 \pi )`

    Returns:
        np.ndarray: The constructed Squeezing matrix representing the Fock operator.
    """

    sechr = 1.0 / np.cosh(r)
    A = np.exp(1j * phi) * np.tanh(r)
    Aconj = np.conj(A)
    sqrt_indices = np.sqrt(np.arange(cutoff))
    sechr_sqrt_indices = sechr * sqrt_indices
    A_conj_sqrt_indices = Aconj * sqrt_indices

    first_row_nonzero = np.sqrt(_double_factorial_array(cutoff)) * np.power(
        -A, np.arange(0, (cutoff + 1) // 2)
    )

    first_row = np.zeros(shape=cutoff, dtype=complex_dtype)
    first_row[np.arange(0, cutoff, 2)] = first_row_nonzero

    roll_index = np.arange(-1, cutoff - 1)
    second_row = sechr_sqrt_indices * first_row[roll_index]

    columns = [first_row, second_row]

    for col in range(2, cutoff):
        columns.append(
            (
                sechr_sqrt_indices * columns[-1][roll_index]
                + A_conj_sqrt_indices[col - 1] * columns[-2]
            )
            / sqrt_indices[col]
        )

    return np.sqrt(sechr) * np.column_stack(columns)
