#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of backends."""

import numpy as np

from piquasso.operator import BaseOperator


class Backend:
    def __init__(self, state):
        """
        Args:
            state (State): The initial quantum state.
        """
        self.state = state

    def execute_instructions(self, instructions):
        """Executes the collected instructions in order.

        Args:
            instructions (list): The methods, parameters and modes of the
                current backend to be executed in order.
        """
        for instruction in instructions:
            operation = instruction['op']
            params = instruction['params']
            modes = instruction['modes']
            operation(self, params, modes)


class FockBackend(Backend):

    def beamsplitter(self, params, modes):
        """Applies a beamsplitter.

        TODO: Multiple particles are not handled yet.

        Args:
            params [float]:
            modes [int]: modes to operate on

        Returns:
            (numpy.ndarray): The representation of the one-particle
                beamsplitter gate on modes `i` and `j`.
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
