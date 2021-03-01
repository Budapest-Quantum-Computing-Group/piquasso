#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.backend import Backend

from piquasso import operations


class GaussianBackend(Backend):

    def get_operation_map(self):
        return {
            operations.PassiveTransform.__name__: self._apply_passive,
            operations.B.__name__: self._apply_passive,
            operations.R.__name__: self._apply_passive,
            operations.GaussianTransform.__name__: self._apply,
            operations.S.__name__: self._apply,
            operations.P.__name__: self._apply,
            operations.S2.__name__: self._apply,
            operations.D.__name__: self.displacement
        }

    def _apply_passive(self, operation):
        self.state.apply_passive(
            operation._passive_representation,
            operation.modes
        )

    def _apply(self, operation):
        self.state.apply_active(
            P=operation._passive_representation,
            A=operation._active_representation,
            modes=operation.modes
        )

    def displacement(self, operation):
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

        :attr:`GaussianState.m` is defined by

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
        """

        params = operation.params
        modes = operation.modes

        if len(params) == 1:
            alpha = params[0]
        else:
            r, phi = params
            alpha = r * np.exp(1j * phi)

        mode = modes[0]

        mean_copy = np.copy(self.state.m)

        self.state.m[mode] += alpha

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
