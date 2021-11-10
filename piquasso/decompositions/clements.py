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

"""Implementation for the Clements decomposition.

References
~~~~~~~~~~

William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf,
W. Steven Kolthammer, Ian A. Walmsley, "An Optimal Design for Universal Multiport
Interferometers", `arXiv:1603.08788 <https://arxiv.org/abs/1603.08788>`_.
"""
from typing import List

import numpy as np


class T(np.ndarray):
    """A definition for the representation of the beamsplitter.

    The matrix is automatically embedded in a `d` times `d` identity
    matrix, which can be readily applied during decomposition.
    """

    def __new__(cls, operation: dict, d: int) -> "T":
        """
        Args:
            operation (dict):
                A dict containing the angle parameters and the modes on which the
                beamsplitter operation is applied.
            d (int): The total number of modes.
        """

        theta, phi = operation["params"]
        i, j = operation["modes"]

        matrix = np.array(
            [
                [np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
                [np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
            ]
        )

        self = np.asarray(np.identity(d, dtype=complex))

        self[i, i] = matrix[0, 0]
        self[i, j] = matrix[0, 1]
        self[j, i] = matrix[1, 0]
        self[j, j] = matrix[1, 1]

        return self.view(cls)

    @classmethod
    def transposed(cls, operation: dict, d: int) -> "T":
        """Transposed beamsplitter matrix.

        Args:
            operation (dict):
                A dict containing the angle parameters and the modes on which the
                beamsplitter operation is applied.
            d (int): The total number of modes.

        Returns:
            T: The transposed beamsplitter matrix.
        """

        theta, phi = operation["params"]

        return cls.transpose(
            cls({"params": (theta, -phi), "modes": operation["modes"]}, d=d)
        ).view(T)

    @classmethod
    def i(cls, operation: dict, d: int) -> "T":
        """Shorthand for :meth:`transposed`.

        The inverse of the matrix equals the transpose in this case.

        Returns:
            T: The transposed beamsplitter matrix.
        """
        return cls.transposed(operation, d)


class Clements:
    def __init__(self, U: np.ndarray, decompose: bool = True):
        """
        Args:
            U (numpy.ndarray): The (square) unitary matrix to be decomposed.
            decompose (bool):
                Optional, if `True`, the decomposition is automatically calculated.
                Defaults to `True`.
        """

        self.U: np.ndarray = U
        self.d: int = U.shape[0]
        self.inverse_operations: List[dict] = []
        self.direct_operations: List[dict] = []
        self.diagonals: np.ndarray

        if decompose:
            self.decompose()

    def decompose(self) -> None:
        """
        Decomposes the specified unitary matrix by application of beamsplitters
        prescribed by the decomposition.
        """

        for column in reversed(range(0, self.d - 1)):
            if column % 2 == 0:
                self.apply_direct_beamsplitters(column)
            else:
                self.apply_inverse_beamsplitters(column)

        self.diagonals = np.diag(self.U)

    def apply_direct_beamsplitters(self, column: int) -> None:
        """
        Calculates the direct beamsplitters for a given column `column`, and
        applies it to `U`.

        Args:
            column (int): The current column.
        """

        for j in range(self.d - 1 - column):
            operation = self.eliminate_lower_offdiagonal(column + j + 1, j)
            self.direct_operations.append(operation)

            beamsplitter = T(operation, d=self.d)

            self.U = beamsplitter @ self.U

    def apply_inverse_beamsplitters(self, column: int) -> None:
        """
        Calculates the inverse beamsplitters for a given column `column`, and
        applies it to `U`.

        Args:
            column (int): The current column.
        """

        for j in reversed(range(self.d - 1 - column)):
            operation = self.eliminate_upper_offdiagonal(column + j + 1, j)
            self.inverse_operations.append(operation)

            beamsplitter = T.i(operation, d=self.d)
            self.U = self.U @ beamsplitter

    def eliminate_lower_offdiagonal(self, i: int, j: int) -> dict:
        """
        Calculates the parameters required to eliminate the lower triangular
        element `i`, `j` of `U` using `T`.
        """
        r = -self.U[i, j] / self.U[i - 1, j]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

        return {
            "modes": (i - 1, i),
            "params": (theta, phi),
        }

    def eliminate_upper_offdiagonal(self, i: int, j: int) -> dict:
        """
        Calculates the parameters required to eliminate the upper triangular
        `i`, `j` of `U` using `T.transposed`.
        """
        r = self.U[i, j] / self.U[i, j + 1]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

        return {
            "modes": (j, j + 1),
            "params": (theta, phi),
        }

    @staticmethod
    def from_decomposition(decomposition: "Clements") -> np.ndarray:
        """
        Creates the unitary operator from the Clements operations.
        """
        U = np.identity(decomposition.d, dtype=complex)

        for operation in decomposition.inverse_operations:
            beamsplitter = T(operation, d=decomposition.d)
            U = beamsplitter @ U

        U = np.diag(decomposition.diagonals) @ U

        for operation in reversed(decomposition.direct_operations):
            beamsplitter = T.i(operation, d=decomposition.d)
            U = beamsplitter @ U

        return U
