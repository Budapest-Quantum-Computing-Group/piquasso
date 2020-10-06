#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.context import Context
from piquasso import constants

from .transformations import quad_transformation


class GaussianState:
    """Object to represent a Gaussian state."""

    def __init__(self, C, G, mean):
        r"""
        A Gaussian state is fully characterised by its mean and covariance
        matrix, i.e. its first and second moments with the quadrature
        operators.

        However, for computations, we only use the
        :math:`C, G \in \mathbb{C}^{d \times d}`
        and the :math:`m \in \mathbb{C}^d` vector.

        Args:
            C (numpy.array): The matrix which is defined by

                .. math::
                    \langle \hat{C}_{ij} \rangle_{\rho} =
                    \langle \hat{a}^\dagger_i \hat{a}_j \rangle_{\rho}.

            G (numpy.array): The matrix which is defined by

                .. math::
                    \langle \hat{G}_{ij} \rangle_{\rho} =
                    \langle \hat{a}_i \hat{a}_j \rangle_{\rho}.

            mean (numpy.array): The vector which is defined by

                .. math::
                    m = \langle \hat{a}_i \rangle_{\rho}.

        """

        self.C = C
        self.G = G
        self.mean = mean

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
        _xp_mean[:self.d] = self.mean.real / np.sqrt(2 * self.hbar)
        _xp_mean[self.d:] = self.mean.imag / np.sqrt(2 * self.hbar)
        return _xp_mean

    @property
    def xp_corr(self):
        r"""The state's correlation matrix in the xp basis.

        Let :math:`\sigma_{(xp)}` be the correlation matrix in the xp basis.
        Then

        .. math::
            \sigma_{ij (xp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho,

        where :math:`\sigma_{ij (xp)}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is a density operator.

        Returns:
            np.array: The :math:`d \times d` covariance matrix in the xp basis.
        """

        d = self.d

        corr = np.empty((2*d, 2*d), dtype=complex)

        C = self.C
        G = self.G

        corr[:d, :d] = 2 * (G + C).real + np.identity(d)
        corr[:d, d:] = 2 * (G + C).imag
        corr[d:, d:] = 2 * (G - C).real - np.identity(d)
        corr[d:, :d] = 2 * (G - C).imag

        return corr * self.hbar

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
    def sigma(self):
        r"""Returns the quadrature-ordered correlation matrix of the state.

        Let :math:`\sigma` be the correlation matrix in the quadrature basis.
        Then

        .. math::
            \sigma_{ij} = \langle R_i R_j + R_j R_i \rangle_\rho,

        where :math:`\sigma_{ij}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is a density operator.

        Returns:
            np.array: The :math:`2d \times 2d` quad-ordered covariance matrix.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_corr @ T.transpose()

    @property
    def quad_representation(self):
        r"""The state's mean and covariance matrix ordered by the quadrature basis.

        Returns:
            tuple: :meth:`mu`, :meth:`sigma`.
        """

        return self.mu, self.sigma
