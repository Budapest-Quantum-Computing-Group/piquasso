#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of the Fock backends."""

import numpy as np

from piquasso.backend import Backend
from piquasso.operator import BaseOperator


class FockBackend(Backend):

    def phaseshift(self, params, modes):
        raise NotImplementedError()

    def beamsplitter(self, params, modes):
        """Applies a beamsplitter.

        TODO: Multiple particles are not handled yet.

        Args:
            params ([float]):
            modes ([int]): modes to operate on

        Returns:
            (numpy.ndarray): The representation of the one-particle
                beamsplitter operation on modes `i` and `j`.
        """

        theta, phi = params
        i, j = modes

        t = np.cos(theta)
        r = np.exp(1j * phi) * np.sin(theta)

        matrix = np.array([[t, r], [-r.conj(), t]])

        d = self.state.d

        embedded_matrix = np.asarray(np.identity(d, dtype=complex))

        embedded_matrix[i, i] = matrix[0, 0]
        embedded_matrix[i, j] = matrix[0, 1]
        embedded_matrix[j, i] = matrix[1, 0]
        embedded_matrix[j, j] = matrix[1, 1]

        BaseOperator(embedded_matrix).apply(self.state)
