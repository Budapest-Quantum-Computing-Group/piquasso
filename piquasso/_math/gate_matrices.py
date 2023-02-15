#
# Copyright 2021-2023 Budapest Quantum Computing Group
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
    fock_indices = np.sqrt(np.arange(cutoff), dtype=complex)
    displacement = r * np.exp(1j * phi)
    transformation = np.zeros((cutoff,) * 2, dtype=complex)
    transformation[0, 0] = np.exp(-0.5 * r**2)
    for row in range(1, cutoff):
        transformation[row, 0] = (
            displacement / fock_indices[row] * transformation[row - 1, 0]
        )
    for row in range(cutoff):
        for col in range(1, cutoff):
            transformation[row, col] = (
                -np.conj(displacement)
                / fock_indices[col]
                * transformation[row, col - 1]
            ) + (
                fock_indices[row] / fock_indices[col] * transformation[row - 1, col - 1]
            )

    return transformation


def create_single_mode_squeezing_matrix(
    r: float,
    phi: float,
    cutoff: int,
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

    transformation = np.zeros((cutoff,) * 2, dtype=complex)
    transformation[0, 0] = np.sqrt(sechr)

    fock_indices = np.sqrt(np.arange(cutoff, dtype=complex))

    for index in range(2, cutoff, 2):
        transformation[index, 0] = (
            -fock_indices[index - 1]
            / fock_indices[index]
            * (transformation[index - 2, 0] * A)
        )

    for row in range(0, cutoff):
        for col in range(1, cutoff):
            if (row + col) % 2 == 0:
                transformation[row, col] = (
                    1
                    / fock_indices[col]
                    * (
                        (fock_indices[row] * transformation[row - 1, col - 1] * sechr)
                        + (
                            fock_indices[col - 1]
                            * A.conj()
                            * transformation[row, col - 2]
                        )
                    )
                )

    return transformation
