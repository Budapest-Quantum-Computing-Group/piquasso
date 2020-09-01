#
# Copyright (C) 2020 by TODO - All rights reserved.
#


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
    def d(self):
        """The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.mean)
