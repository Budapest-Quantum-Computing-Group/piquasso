# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso import constants
from piquasso.context import Context
from piquasso.state import State

from .transformations import quad_transformation


class GaussianState(State):
    r"""Object to represent a Gaussian state.

    Attributes:
        mean (numpy.array): The expectation value of the annihilation operators on all
            modes (a vector, essentially), and is defined by

            .. math::
                m = \langle \hat{a}_i \rangle_{\rho}.

        C (numpy.array): A correlation matrix which is defined by

            .. math::
                \langle \hat{C}_{ij} \rangle_{\rho} =
                \langle \hat{a}^\dagger_i \hat{a}_j \rangle_{\rho}.

        G (numpy.array): A correlation matrix which is defined by

                .. math::
                    \langle \hat{G}_{ij} \rangle_{\rho} =
                    \langle \hat{a}_i \hat{a}_j \rangle_{\rho}.

    """

    def __init__(self, C, G, mean):
        r"""
        A Gaussian state is fully characterised by its mean and correlation
        matrix, i.e. its first and second moments with the quadrature
        operators.

        However, for computations, we only use the
        :math:`C, G \in \mathbb{C}^{d \times d}`
        and the :math:`m \in \mathbb{C}^d` vector.

        Args:
            C (numpy.array): See :attr:`C`.
            G (numpy.array): See :attr:`G`.
            mean (numpy.array): See :attr:`mean`.

        """

        self.C = C
        self.G = G
        self.mean = mean

    @classmethod
    def create_vacuum(cls, d):
        r"""Creates a Gaussian vacuum state.

        Args:
            d (int): The number of modes.

        Returns:
            GaussianState: A Gaussan vacuum state.
        """

        return cls(
            C=np.zeros((d, d), dtype=complex),
            G=np.zeros((d, d), dtype=complex),
            mean=np.zeros(d, dtype=complex),
        )

    @property
    def hbar(self):
        """Reduced Plack constant.

        TODO: It would be better to move this login into
        :mod:`piquasso.context` after a proper context implementation.

        Returns:
            float: The value of the reduced Planck constant.
        """
        if Context.current_program:
            return Context.current_program.hbar

        return constants.HBAR_DEFAULT

    @property
    def d(self):
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.mean)

    @property
    def xp_mean(self):
        r"""The state's mean in the xp basis.

        Returns:
            np.array: A :math:`d`-vector.
                The expectation value of the quadrature operators in xp basis,
                i.e. :math:`\operatorname{Tr} \rho \hat{Y}` , where
                :math:`\hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.
        """

        _xp_mean = np.empty(2 * self.d)
        _xp_mean[:self.d] = self.mean.real * np.sqrt(2 * self.hbar)
        _xp_mean[self.d:] = self.mean.imag * np.sqrt(2 * self.hbar)
        return _xp_mean

    @property
    def xp_corr(self):
        r"""The state's correlation matrix in the xp basis.

        Let :math:`M_{(xp)}` be the correlation matrix in the xp basis.
        Then

        .. math::
            M_{ij (xp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho,

        where :math:`M_{ij (xp)}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` correlation matrix in the xp basis.
        """

        d = self.d

        corr = np.empty((2*d, 2*d), dtype=complex)

        C = self.C
        G = self.G

        corr[:d, :d] = 2 * (G + C).real + np.identity(d)
        corr[:d, d:] = 2 * (G + C).imag
        corr[d:, d:] = 2 * (-G + C).real + np.identity(d)
        corr[d:, :d] = 2 * (G - C).imag

        return corr * self.hbar

    @property
    def xp_cov(self):
        r"""The xp-ordered coveriance matrix of the state.

        The xp-ordered covariance matrix :math:`\sigma_{xp}` is defined by

        .. math::
            \sigma_{xp, ij} := \langle Y_i Y_j + Y_j Y_i \rangle_\rho
                - 2 \langle Y_i \rangle_\rho \langle Y_j \rangle_\rho,

        where

        .. math::
            \hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` xp-ordered covariance matrix in xp basis.
        """
        xp_mean = self.xp_mean
        return self.xp_corr - 2 * np.outer(xp_mean, xp_mean)

    @property
    def xp_representation(self):
        r"""
        The state's mean and correlation matrix ordered in the xp basis.

        Returns:
            tuple: :meth:`xp_mean`, :meth:`xp_corr`.
        """

        return self.xp_mean, self.xp_corr

    @property
    def mu(self):
        r"""Returns the xp-ordered mean of the state.

        Returns:
            np.array: A :math:`2d`-vector.
                The expectation value of the quadrature operators in
                xp-ordering, i.e. :math:`\operatorname{Tr} \rho \hat{R}`, where
                :math:`\hat{R} = (x_1, p_1, \dots, x_d, p_d)^T`.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_mean

    @property
    def corr(self):
        r"""Returns the quadrature-ordered correlation matrix of the state.

        Let :math:`M` be the correlation matrix in the quadrature basis.
        Then

        .. math::
            M_{ij} = \langle R_i R_j + R_j R_i \rangle_\rho,

        where :math:`M_{ij}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` quad-ordered correlation matrix.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_corr @ T.transpose()

    @property
    def cov(self):
        r"""The quadrature-ordered coveriance matrix of the state.

        The quadrature-ordered covariance matrix :math:`\sigma` is defined by

        .. math::
            \sigma_{ij} := \langle R_i R_j + R_j R_i \rangle_\rho
                - 2 \langle R_i \rangle_\rho \langle R_j \rangle_\rho,

        where

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` quadrature-ordered covariance matrix in
                xp basis.
        """
        mu = self.mu
        return self.corr - 2 * np.outer(mu, mu)

    @property
    def quad_representation(self):
        r"""The state's mean and correlation matrix ordered by the quadrature basis.

        Returns:
            tuple: :meth:`mu`, :meth:`corr`.
        """

        return self.mu, self.corr

    def rotated(self, phi):
        r"""Returns the copy of the current state, rotated by `phi`.

        Let :math:`\phi \in [ 0, 2 \pi )`. Let us define the following:

        .. math::
            x_{i, \phi} = \cos\phi~x_i + \sin\phi~p_i,

        which is a generalized quadrature operator. One could rotate the whole state by
        this simple, phase space transformation.

        Using the transformation rules between the ladder operators and quadrature
        operators, i.e.

        .. math::
            x_i &= \sqrt{\frac{\hbar}{2}} (a_i + a_i^\dagger) \\
            p_i &= -i \sqrt{\frac{\hbar}{2}} (a_i - a_i^\dagger),

        we could rewrite :math:`x_{i, \phi}` to the following form:

        .. math::
            x_{i, \phi} = \sqrt{\frac{\hbar}{2}} \left(
                a_i \exp(-i \phi) + a_i^\dagger \exp(i \phi)
            \right)

        which means, that e.g. the annihilation operators `a_i` are transformed just
        multiplied by a phase factor :math:`\exp(-i \phi)` under this phase space
        rotation, i.e.

        .. math::
            (\langle a_i \rangle_{\rho} =: )~m_i &\mapsto \exp(-i \phi) m_i \\
            (\langle a^\dagger_i a_j \rangle_{\rho} =: )~C_{ij} &\mapsto C_{ij} \\
            (\langle a_i a_j \rangle_{\rho} =: )~G_{ij} &\mapsto \exp(-i 2 \phi) G_{ij}.

        Args:
            phi (float): The angle to rotate the state with.

        Returns:
            GaussianState: The rotated `GaussianState` instance.
        """
        phase = np.exp(- 1j * phi)

        return GaussianState(
            C=self.C,
            G=(self.G * phase**2),
            mean=(self.mean * phase),
        )

    def reduced(self, modes):
        """Returns the copy of the current state, reduced to the given `modes`.

        This method essentially preserves the modes specified from the representation
        of the Gaussian state, but cuts out the other modes.

        Args:
            modes (tuple): The modes to reduce the state to.

        Returns:
            GaussianState: The reduced `GaussianState` instance.
        """
        return GaussianState(
            C=self.C[np.ix_(modes, modes)],
            G=self.G[np.ix_(modes, modes)],
            mean=self.mean[np.ix_(modes)],
        )

    def reduced_rotated_mean_and_cov(self, modes, phi):
        r"""The quadrature operator's mean and covariance on a rotated and reduced state.

        Let the index set :math:`\vec{i}` correspond to `modes`, and the angle
        :math:`\phi` correspond to `phi`. The current :class:`GaussianState` instance
        is reduced to `modes` and rotated by `phi` in a new instance, and let that
        state be denoted by :math:`\rho_{\vec{i}, \phi}`.

        Then the quadrature ordered mean and covariance can be calculated by

        .. math::
            \mu_{\vec{i}, \phi}
                &:= \langle \hat{R}_{\vec{i}} \rangle_{\rho_{\vec{i}, \phi}}, \\
            \sigma_{\vec{i}, \phi}
                &:=  \langle
                    \hat{R}_{\vec{i}} \hat{R}_{\vec{i}}^T
                \rangle_{\rho_{\vec{i}, \phi}}
                - \mu_{\vec{i}, \phi} \mu_{\vec{i}, \phi}^T,

        where

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\hat{R}_{\vec{i}}` is just the same vector, reduced to a subsystem
        specified by :math:`\vec{i}`.

        Args:
            modes (tuple): The modes to reduce the state to.
            phi (float): The angle to rotate the state with.

        Returns:
            tuple:
                Quadrature ordered mean and covariance of the reduced and rotated
                version of the current :class:`GaussianState`.
        """
        transformed_state = self.reduced(modes).rotated(phi)

        return transformed_state.mu, transformed_state.cov

    def apply(self, T, modes):
        r"""Apply a transformation to the quantum state.

        Let :math:`\vec{m}` denote an index set, which corresponds to the parameter
        `modes`.

        Let :math:`T \in \mathbb{C}^{k \times k},\, k \in [d]` be a transformation
        which transforms the vector of annihilation operators in the following manner:

        .. math::
            \mathbf{a}_{\vec{m}} \mapsto T \mathbf{a}_{\vec{m}},

        or in terms of vector elements:

        .. math::
            a_{i} \mapsto \sum_{j \in \vec{m}} T^{ij} a_j

        Application to :attr:`mean` is done by matrix multiplication.

        The canonical commutation relations can be written as

        .. math::
            [a^\dagger_i, a_j] = \delta_{i j},

        and then applying the transformation :math:`T` we get

        .. math::
            \sum_{i, j \in \vec{m}} [T^*_{ki} a^\dagger_i, T_{lj} a_j]
                &= \sum_{i, j \in \vec{m}} T^*_{ki} T_{lj}
                    [a^\dagger_i, a_j] \\
                &= \sum_{i, j \in \vec{m}} T^*_{ki} T_{lj} \delta_{i j} \\
                &= \sum_{i \in \vec{m}} T^*_{ki} T_{li} \\
                &= \sum_{i \in \vec{m}} (T^\dagger)_{ik} T_{li} \\
                &= \delta_{k l},

        where the last line imposes, that any transformation should leave the canonical
        commutation relations invariant.
        The last line of the equation means, that :math:`T` should actually be a
        unitary matrix.

        Application to `C` and `G` is non-trivial however: one has to apply the
        transformation for the external modes as well, see :meth:`apply_to_C_and_G`.

        Args:
            T (numpy.array): The matrix to be applied.
            modes (tuple): The modes, on which the matrix should operate.
        """

        self.mean[modes, ] = T @ self.mean[modes, ]

        self.apply_to_C_and_G(T, modes=modes)

    def apply_to_C_and_G(self, T, modes):
        r"""Applies the matrix :math:`T` to the :math:`C` and :math:`G`.

        Let :math:`\vec{i}` denote an index set, which corresponds to `index`
        in the implementation. E.g. for 2 modes denoted by :math:`n` and
        :math:`m`:, one could write

        .. math::
                \vec{i} = \{n, m\} \times \{n, m\}.

        From now on, I will use the notation
        :math:`\{n, m\} := \mathrm{modes}`.

        The transformation by :math:`T` can be prescribed in the following
        manner:

        .. math::
                C_{\vec{i}} \mapsto T^* C_{\vec{i}} T^T \\
                G_{\vec{i}} \mapsto T G_{\vec{i}} T^T

        If there are other modes in the system, i.e. `modes` does not refer to all the
        modes, :meth:`_apply_to_other_modes` is called to handle those.

        Note:
            For indexing of numpy arrays, see
            https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing

        Args:
            T (np.array): The matrix to be applied.
            modes (tuple): The modes, on which the transformation should directly
                operate.
        """

        transformed_columns = np.array([modes] * len(modes))
        transformed_rows = transformed_columns.transpose()

        index = transformed_rows, transformed_columns

        self.C[index] = T.conjugate() @ self.C[index] @ T.transpose()
        self.G[index] = T @ self.G[index] @ T.transpose()

        other_modes = np.delete(np.arange(self.d), modes)

        if other_modes.size != 0:
            self._apply_to_other_modes(T, modes, other_modes)

    def _apply_to_other_modes(self, T, modes, other_modes):
        r"""Applies the matrix :math:`T` to modes which are not directly transformed.

        This method is applied for the correlation matrices :math:`C` and :math:`G`.
        For context, visit :meth:`apply_to_C_and_G`.

        Let us denote :math:`\vec{k}` the following:

        .. math::
                \vec{k} = \mathrm{modes}
                        \times \big (
                                [d]
                                - \mathrm{modes}
                        \big ).

        For all the remaining modes, the following is applied regarding the
        elements, where the **first** index corresponds to
        :math:`\mathrm{modes}`:

        .. math::
                C_{\vec{k}} \mapsto T^* C_{\vec{k}} \\
                G_{\vec{k}} \mapsto T G_{\vec{k}}

        Regarding the case where the **second** index corresponds to
        :math:`\mathrm{modes}`, i.e. where we use
        :math:`\big ( [d] - \mathrm{modes} \big )
        \times \mathrm{modes}`, the same has to be applied.

        For :math:`n \in \mathrm{modes}` and :math:`m \in [d]`, we could
        use

        .. math::
                C_{nm} := C^*_{mn} \\
                G_{nm} := G_{mn}.


        Args:
            T (np.array): The matrix to be applied.
            modes (tuple): The modes, on which the transformation should directly
                operate.
            other_modes (tuple): The modes, on which the transformation is not directly
                applied, but should be accounted for in :math:`C` and :math:`G`.

        """
        other_rows = np.array([modes] * len(other_modes)).transpose()

        index = other_rows, other_modes

        self.C[index] = T.conjugate() @ self.C[index]
        self.G[index] = T @ self.G[index]

        self.C[:, modes] = np.conj(self.C[modes, :]).transpose()
        self.G[:, modes] = self.G[modes, :].transpose()

    def apply_active(self, alpha, beta, modes):
        r"""
        This method updates the vector of the means when an active operation such as `Squeezing` is applied
        and calls :meth:`apply_active_to_C_and_G`.

        The vector of the means of the :math:`ith` mode:
        :math:`m_i = \langle \hat{a}_i \rangle_\rho` is evolved as follows:

        .. math::
            {S(z)}^{\dagger}m_i S(z) = \alpha\hat{a} - \beta\hat{a}^\dagger \\
                = \alpha m_i - \beta m_i^*

        Args:
            alpha (complex): A complex that represents the value of :math:`cosh(amp)`.
            beta (complex): A complex that represents the value of :math:`e^{i\theta}\sinh(amp)`.
            modes (tuple): The qumode index on which the squeezing gate operates,
                embedded in a `tuple`.
        """  # noqa: E501
        self.mean[modes[0]] = (alpha * self.mean[modes[0]]) - (
            beta * np.conj(self.mean[modes[0]])
        )
        self.apply_active_to_C_and_G(alpha, beta, modes)

    def apply_active_to_C_and_G(self, alpha, beta, modes):
        r"""
        This method updates the :math:`G_{ij}` and the :math:`C_{ij}` elements and calls :meth:`_apply_active_to_other_modes`.

        By performing an active operation such as `Squeezing` a gaussian state, the element :math:`G_{ij}`
        evolves to be:

        .. math::
            \hat{S}^\dagger(z)\hat{a}\hat{a}\hat{S}(z) = \alpha^2 G_{i j} -
                \alpha\beta - 2\alpha\beta C_{i j} + \beta^2 G_{i j}^\dagger,

        for the :math:`C_{ij}` element, it evolves into:

        .. math::
            \hat{S}^\dagger(z)\hat{a}^\dagger\hat{a}\hat{S}(z) = \alpha^2 C_{ij} -
                \alpha\beta G_{ij}^\dagger - \alpha\beta^\dagger G_{ij} + \beta\beta^* + \beta\beta^* C_{ij}

        Args:
            alpha (complex): A float that represents the value of :math:`\cosh(amp)`.
            beta (complex): A complex that represents the value of :math:`e^{i\theta}\sinh(amp)`.
            modes (tuple): The qumode index on which the squeezing gate operates,
                embedded in a `tuple`.
        """  # noqa: E501
        alpha2 = alpha * np.conj(alpha)
        alpha_beta = alpha * beta

        transformed_index_C = self.C[modes, modes]
        transformed_index_G = self.G[modes, modes]

        self.G[modes, modes] = (
            (alpha2 * transformed_index_G)
            - (alpha_beta)
            - (2 * alpha_beta * transformed_index_C)
            + (beta**2 * np.conj(transformed_index_G))
        )
        self.C[modes, modes] = (
            (alpha2 * transformed_index_C)
            + (beta * np.conj(beta) * transformed_index_C)
            + beta * np.conj(beta)
            - (alpha_beta * np.conj(transformed_index_G))
            - (np.conj(alpha_beta) * transformed_index_G)
        )

        other_modes = np.delete(np.arange(self.d), modes)

        if other_modes.size != 0:
            self._apply_active_to_other_modes(alpha, beta, modes, other_modes)

    def _apply_active_to_other_modes(self, alpha, beta, modes, other_modes):
        r"""
        This method updates the off diagonal elements of the :math:`G` and the :math:`C` matrices.

        The columns :math:`j` defined in `other_modes` associated with the mode :math:`i` defioned in
        `modes` evolve according to the linear transformation defined in :meth:`apply_active`.

        Then each row of the mode :math:`i` will be updated according to the fact that :math:`C_{ij} = C_{ij}^*`
        and :math:`G_{ij} = G_{ij}^T`.

        Args:
            alpha (complex): A complex that represents the value of :math:`cosh(amp)`.
            beta (complex): A complex that represents the value of :math:`e^{i\theta}\sinh(amp)`.
            modes (tuple): The qumode index on which the squeezing gate operates,
                embedded in a `tuple`.
            other_modes (np.array): A vector that contains The modes, on which the transformation
                is not directly applied, but should be accounted for in :math:`C` and :math:`G`.
        """  # noqa: E501
        transformed_index_C = self.C[modes, other_modes]
        transformed_index_G = self.G[modes, other_modes]

        self.C[modes, other_modes] = (alpha * transformed_index_C) - (
            np.conj(beta) * transformed_index_G
        )

        self.G[modes, other_modes] = (alpha * transformed_index_G) - (
            beta * transformed_index_C
        )

        self.C[:, modes] = self.C[modes, :].conj().T

        self.G[:, modes] = self.G[modes, :].T


