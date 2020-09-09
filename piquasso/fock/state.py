#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""A simple quantum state implementation based on numpy."""

import numpy as np

from piquasso.operator import BaseOperator
from piquasso.fock.backend import FockBackend


class FockState(BaseOperator):
    r"""
    Implements the density operator from quantum mechanics in Fock
    representation.
    """

    backend_class = FockBackend

    def __new__(cls, representation):
        return np.array(representation).view(cls)

    @classmethod
    def from_state_vector(cls, state_vector):
        """Creates a density operator from a state vector."""
        return np.outer(state_vector, state_vector).view(cls)

    @property
    def d(self):
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return self.shape[0]
