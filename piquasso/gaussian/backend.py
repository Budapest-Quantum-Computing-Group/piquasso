#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.backend import Backend


class GaussianBackend(Backend):
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

    def beamsplitter(self, params, modes):
        r"""Applies the beamsplitter gate to the state.

        The matrix representation of the beamsplitter operation
        is

        .. math::
            B = \begin{bmatrix}
                t  & r^* \\
                -r & t
            \end{bmatrix},

        where :math:`t = \cos(\theta)` and
        :math:`r = e^{- i \phi} \sin(\theta)`.

        Args:
            params (tuple): Angle parameters :math:`\phi` and :math:`\theta` for the
                beamsplitter gate.
            modes (tuple): Distinct positive integer values which are used to represent
                qumodes.
        """

        phi, theta = params

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        B = np.array(
            [
                [ t, np.conj(r)],
                [-r,          t]
            ]
        )

        self.state.mean[modes, ] = B @ self.state.mean[modes, ]

        self.state.apply_to_C_and_G(B, modes=modes)
