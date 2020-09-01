#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.backend import Backend


class GaussianBackend(Backend):

    def beamsplitter(self, params, modes):
        raise NotImplementedError()

    def phaseshift(self, params, modes):
        """Performs a phase shifting on the quantum state.

        The annihilation and creation operators are evolved in the following
        way:

        .. math::
            P(\phi) \hat{a}_k P(\phi)^\dagger = e^{i \phi} \hat{a}_k \\
            P(\phi) \hat{a}_k^\dagger P(\phi)^\dagger
                = e^{- i \phi} \hat{a}_k^\dagger

        Args:
            params (tuple): An iterable with a single element, which
                corresponds to the angle of the phaseshifter.
            modes (tuple): An iterable with a single element, which
                corresponds to the mode of the phaseshifter.
        """
        phi = params[0]
        k = modes[0]

        phase = np.exp(1j * phi)

        self.state.mean[k] *= phase

        self.state.G[k][k] *= phase ** 2

        idx = np.delete(np.arange(self.state.d), k)
        self.state.C[k][idx] *= np.conj(phase)
        self.state.G[k][idx] *= phase

        self.state.C[:, k] = np.conj(self.state.C[k, :])
        self.state.G[:, k] = self.state.G[k, :]
