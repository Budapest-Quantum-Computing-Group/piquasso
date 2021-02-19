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

        phase = np.exp(1j * phi)

        P = np.array(
            [
                [phase],
            ]
        )

        self.state.apply_passive(P, modes)

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

        self.state.apply_passive(B, modes)

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

        The :math:`\hat{a}_i \mapsto hat{a}_i + \alpha \mathbb{1}` transformation
        should be applied to the `GaussianState.C` and `GaussianState.G` matrices as
        well:

        .. math::
            C_{ii} \mapsto C_{ii} + \alpha m_i^* + \alpha^* m_i + | \alpha |^2, \\
            G_{ii} \mapsto G_{ii} + 2 \alpha m_i + \alpha^2, \\

        and to the corresponding `i, j (i \neq j)` modes.

        Note, that :math:`\alpha` is often written in the form

        .. math:
            \alpha = r \exp(i \phi),

        where :math:`r \geq 0` and :math:`\phi \in [ 0, 2 \pi )`. When two parameters
        are specified for this operation, the first is interpreted as :math:`r`, and the
        second one as :math:`\phi`.

        Also note, that the displacement cannot be categorized as an active or passive
        linear transformation, because the unitary transformation does not strictly
        produce a linear combination of the field operators.

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

        mean_copy = np.copy(self.state.mean)

        self.state.mean[mode] += alpha

        self.state.C[mode, mode] += (
            alpha * np.conj(mean_copy[mode])
            + np.conj(alpha) * mean_copy[mode]
            + np.conj(alpha) * alpha
        )

        self.state.G[mode, mode] += 2 * alpha * mean_copy[mode] + alpha * alpha

        other_modes = np.delete(np.arange(self.state.d), modes)

        self.state.C[other_modes, mode] += alpha * mean_copy[other_modes]
        self.state.C[mode, other_modes] += np.conj(alpha) * mean_copy[other_modes]

        self.state.G[other_modes, mode] += alpha * mean_copy[other_modes]
        self.state.G[mode, other_modes] += alpha * mean_copy[other_modes]

    def squeezing(self, params, modes):
        r"""
        This method implements the squeezing operator for the gaussian backend. The standard
        definition of its operator is:

        .. math::
                S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

        where :math:`z = r e^{i\theta}`. The :math:`r` parameter is the amplitude of the squeezing
        and :math:`\theta` is the angle of the squeezing.

        This act of squeezing at a given rotation angle :math:`\theta` results in a shrinkage in the
        :math:`\hat{x}` quadrature and a stretching in the other quadrature :math:`\hat{p}` as follows:

        .. math::
            S^\dagger(z) x_{\theta} S(z) = e^{-r} x_{\theta}, \: S^\dagger(z) p_{\theta} S(z) = e^{r} p_{\theta}

        The action of the :math:`\hat{S}(z)` gate on the ladder operators :math:`\hat{a}`
        and :math:`\hat{a}^\dagger` can be defined as follows:

        .. math::
            {S(z)}^{\dagger}\hat{a}S(z) = \alpha\hat{a} - \beta \hat{a}^{\dagger} \\
                {S(z)}^{\dagger}\hat{a}^\dagger S(z) = \alpha\hat{a}^\dagger - \beta^* \hat{a}

        where :math:`\alpha` and :math:`\beta` are :math:`\cosh(amp)`, :math:`e^{i\theta}\sinh(amp)`
        respectively.

        Args:
            params (tuple): The parameters for the squeezing gate are in the form of
                `(amp, )` or `(amp, theta)` where the amplitude is a real float that represents the
                magnitude of the squeezing gate and :math:`\theta` which is in radians :math:`\in [0, 2 \pi)`.
            modes (tuple): The qumode index on which the squeezing gate operates,
                embedded in a `tuple`.
        """  # noqa: E501

        if len(params) == 1:
            theta = 0
        else:
            theta = params[1]

        alpha = np.cosh(params[0]) + 0j

        beta = np.sinh(params[0]) * np.exp(1j * theta)

        P = np.array([[alpha]])

        A = np.array([[- beta]])

        self.state.apply_active(P, A, modes)
