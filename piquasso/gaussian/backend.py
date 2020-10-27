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
        r"""Applies the beamsplitter operation to the state.

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
                beamsplitter operation.
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

    def displacement(self, params, modes):
        r"""Applies the displacement operation to the state.

        .. math::
            D(\alpha) = \exp(\alpha \hat{a}_i^\dagger - \alpha^* \hat{a}_i),

        where :math:`\alpha \in \mathbb{C}` is a parameter, :math:`\hat{a}_i` and
        :math:`\hat{a}_i^\dagger` are the annihilation and creation operators on the
        :math:`i`-th mode, respectively.

        The displacement operation acts on the annihilation and creation operators
        in the following way:

        .. math::
            D^\dagger (\alpha) \hat{a}_i D (\alpha) = \hat{a}_i + \alpha \mathbb{1}.

        `GaussianState.mean` is defined by

        .. math::
            m = \langle \hat{a}_i \rangle_{\rho}.

        By using the displacement, one acquires

        .. math::
            m_{\mathrm{displaced}}
                = \langle D^\dagger (\alpha) \hat{a}_i D (\alpha) \rangle_{\rho}
                = \langle \hat{a}_i + \alpha \mathbb{1}) \rangle_{\rho}
                = m + \alpha.

        Note, that :math:`\alpha` is often written in the form

        .. math:
            \alpha = r \exp(i \phi),

        where :math:`r \geq 0` and :math:`\phi \in [ 0, 2 \pi )`. When two parameters
        are specified for this operation, the first is interpreted as :math:`r`, and the
        second one as :math:`\phi`.

        Args:
            params (tuple): Parameter(s) for the displacement operation in the form of
                `(alpha, )` or `(r, phi)`, where `alpha == r * np.exp(1j * phi)`.
            modes (tuple): The qumode index on which the displacement operation
                operates, embedded in a `tuple`.

        """

        if len(params) == 1:
            alpha = params[0]
        else:
            r, phi = params
            alpha = r * np.exp(1j * phi)

        mode = modes[0]

        self.state.mean[mode] += alpha
