#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""
Implementation for the Clements decomposition

References
----------
.. [1] William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf,
    W. Steven Kolthammer, Ian A. Walmsley, "An Optimal Design for
    Universal Multiport Interferometers", arXiv:1603.08788.
"""

import numpy as np

from piquasso.operator import BaseOperator


class T(BaseOperator):
    """
    A definition for the representation
    of the beamsplitter.
    """

    def __new__(cls, operation, d):
        """Produces the beamsplitter matrix.

        The matrix is automatically embedded in a `d` times `d` identity
        matrix, which can be readily applied during decomposition.

        Args:
            operation (dict): A dict containing the angle parameters and the
                modes on which the beamsplitter operation is applied.
            d (int): The total number of modes.

        Returns:
            T: The beamsplitter matrix.
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

        return self.view(BaseOperator)

    @classmethod
    def transposed(cls, operation, d):
        """Produces the transposed beamsplitter matrix.

        Args:
            operation (dict): A dict containing the angle parameters and the
                modes on which the beamsplitter operation is applied.
            d (int): The total number of modes.

        Returns:
            T: The transposed beamsplitter matrix.
        """

        theta, phi = operation["params"]

        return np.transpose(
            cls({"params": (theta, -phi), "modes": operation["modes"]}, d=d)
        )

    @classmethod
    def i(cls, *args, **kwargs):
        """Shorthand for :meth:`transposed`.

        The inverse of the matrix equals the transpose in this case.

        Returns:
            T: The transposed beamsplitter matrix.
        """
        return cls.transposed(*args, **kwargs)


class Clements:
    def __init__(self, U, decompose=True):
        """
        Args:
            U (numpy.ndarray): The (square) unitary matrix to be decomposed.
            decompose (bool): Optional, if `True`, the decomposition is
                automatically calculated. Defaults to `True`.
        """
        self.U = U
        self.d = U.shape[0]
        self.inverse_operations = []
        self.direct_operations = []
        self.diagonals = None

        if decompose:
            self.decompose()

    def decompose(self):
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

    def apply_direct_beamsplitters(self, column):
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

    def apply_inverse_beamsplitters(self, column):
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

    def eliminate_lower_offdiagonal(self, i, j):
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

    def eliminate_upper_offdiagonal(self, i, j):
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
    def from_decomposition(decomposition):
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

        return BaseOperator(U)
