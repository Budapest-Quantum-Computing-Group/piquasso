#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of operators used in numerical calculations."""

import numpy as np


class BaseOperator(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(cls)

    def apply(self, state):
        """Application of the current operator to the quantum state `state`.

        Args:
            state (State): The initial quantum state.

        Returns:
            (State): The evolved quantum state.
        """
        return (self @ state @ self.adjoint()).view(state.__class__)

    def adjoint(self):
        """
        Returns:
            (BaseOperator): The adjoint of the operator.
        """
        return self.conjugate().transpose().view(self.__class__)

    def is_unitary(self, tol=1e-10):
        """
        Args:
            tol (float, optional): The tolerance for testing the unitarity.
                Defaults to `1e-10`.

        Returns:
            bool: `True` if the current object is unitary within the specified
                tolerance `tol`, else `False`.
        """
        return (self @ self.adjoint() - np.identity(self.shape[0]) < tol).all()
