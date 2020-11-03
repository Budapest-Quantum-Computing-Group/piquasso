#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""A simple quantum state implementation based on numpy."""

from .backend import PassiveGaussianBackend


class PassiveGaussianState:
    r"""Represents the state of a pure gaussian quantum system

    A Gaussian state is fully characterised by its mean and correlation
    matrix, i.e. its first and second moments with the quadrature
    operators.

    However, for computations, we only use the
    :math:`C, G \in \mathbb{C}^{d \times d}`
    and the :math:`m \in \mathbb{C}^d` vector.

    However, since the application of passive operations will not affect $G$
    and the mean vector, the :math:`C` correlation matrix can represent the state alone.
    """

    backend_class = PassiveGaussianBackend

    def __init__(self, C):
        r"""Creates a PassiveGaussianState

        C (numpy.array): The matrix which is defined by
            .. math::
                \langle \hat{C}_{ij} \rangle_{\rho} =
                \langle \hat{a}^\dagger_i \hat{a}_j \rangle_{\rho}.
        """
        self.C = C

    @property
    def d(self):
        """The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return self.C.shape[0]
