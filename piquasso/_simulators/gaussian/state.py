#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, List, Union

from scipy.linalg import block_diag

import numpy as np

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState, InvalidParameter, PiquassoException
from piquasso.api.state import State
from piquasso.api.connector import BaseConnector

from piquasso._math.functions import gaussian_wigner_function
from piquasso._math.linalg import (
    is_symmetric,
    is_positive_semidefinite,
)
from piquasso._math.symplectic import symplectic_form
from piquasso._math.fock import get_fock_space_basis
from piquasso._math.transformations import from_xxpp_to_xpxp_transformation_matrix

from piquasso._math.decompositions import williamson

from .probabilities import (
    DensityMatrixCalculation,
    DisplacedDensityMatrixCalculation,
    NondisplacedDensityMatrixCalculation,
    calculate_click_probability_nondisplaced,
    calculate_click_probability,
)


class GaussianState(State):
    r"""Class to represent a Gaussian state."""

    def __init__(
        self, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.

        """
        super().__init__(connector=connector, config=config)
        self._d = d
        self.reset()

    def __len__(self) -> int:
        return self._d

    @property
    def d(self) -> int:
        return len(self)

    def reset(self) -> None:
        r"""Resets the state to a vacuum."""

        np = self._connector.np

        vector_shape = (self.d,)
        matrix_shape = vector_shape * 2

        self._m = np.zeros(vector_shape, dtype=self._config.complex_dtype)
        self._G = np.zeros(matrix_shape, dtype=self._config.complex_dtype)
        self._C = np.zeros(matrix_shape, dtype=self._config.complex_dtype)

    @classmethod
    def _from_representation(
        cls,
        *,
        m: np.ndarray,
        G: np.ndarray,
        C: np.ndarray,
        config: Config,
        connector: BaseConnector,
    ) -> "GaussianState":
        obj = cls(d=len(m), connector=connector, config=config)

        obj._m = m
        obj._G = G
        obj._C = C

        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianState):
            return False

        return (
            np.allclose(self._C, other._C)
            and np.allclose(self._G, other._G)
            and np.allclose(self._m, other._m)
        )

    def validate(self) -> None:
        r"""
        Validates the state.

        Raises:
            InvalidState:
                Raised if the underlying Gaussian state is invalid, which could mean
                - ill-shaped mean and covariance;
                - non-symmetric covariance matrix;
                - the covariance matrix doesn't fulfill the Robertson-Schrödinger
                  uncertainty relations.
        """
        self._validate_mean(self.xpxp_mean_vector, self.d)
        self._validate_cov(self.xpxp_covariance_matrix, self.d)

    def _validate_mean(self, mean: np.ndarray, d: int) -> None:
        if not self._config.validate:
            return

        expected_shape = (2 * d,)

        if not mean.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'mean' vector shape; "
                f"expected={expected_shape}, actual={mean.shape}."
            )

    def _validate_cov(self, cov: np.ndarray, d: int) -> None:
        if not self._config.validate:
            return

        expected_shape = (2 * d,) * 2

        if not cov.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'cov' matrix shape; "
                f"expected={expected_shape}, actual={cov.shape}."
            )

        if not is_symmetric(cov):
            raise InvalidState("The covariance matrix is not symmetric.")

        if not is_positive_semidefinite(
            cov / self._config.hbar + 1j * symplectic_form(d)
        ):
            raise InvalidState(
                "The covariance matrix is invalid, since it doesn't fulfill the "
                "Robertson-Schrödinger uncertainty relation."
            )

    @property
    def xxpp_mean_vector(self) -> np.ndarray:
        r"""The state's mean in the xxpp-ordered basis.

        The expectation value of the quadrature operators in xxpp basis, i.e.
        :math:`\operatorname{Tr} \left ( \rho Y \right )`, where
        :math:`Y = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.

        Returns:
            numpy.ndarray: A :math:`2d`-vector where :math:`d` is the number of modes.
        """

        np = self._connector.np

        dimensionless_xxpp_mean_vector = np.concatenate(
            [self._m.real, self._m.imag]
        ) * np.sqrt(2)

        return dimensionless_xxpp_mean_vector * np.sqrt(self._config.hbar)

    @xxpp_mean_vector.setter
    def xxpp_mean_vector(self, value):
        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        self.xpxp_mean_vector = T @ value

    @property
    def xxpp_covariance_matrix(self) -> np.ndarray:
        r"""The xxpp-ordered coveriance matrix of the state.

        The xxpp-ordered covariance matrix :math:`\sigma_{xp}` is defined by

        .. math::
            \sigma_{xp, ij} := \langle Y_i Y_j + Y_j Y_i \rangle_\rho
                - 2 \langle Y_i \rangle_\rho \langle Y_j \rangle_\rho,

        where

        .. math::
            Y = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray:
                The :math:`2d \times 2d` xp-ordered covariance matrix.
        """
        np = self._connector.np

        C = self._C
        G = self._G

        dimensionless_xxpp_covariance_matrix = 2 * np.block(
            [
                [(G + C).real, (G + C).imag],
                [(G - C).imag, (-G + C).real],
            ],
        ) + np.identity(2 * self.d)

        return dimensionless_xxpp_covariance_matrix * self._config.hbar

    @xxpp_covariance_matrix.setter
    def xxpp_covariance_matrix(self, value):
        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        self.xpxp_covariance_matrix = T @ value @ T.transpose()

    @property
    def xxpp_correlation_matrix(self) -> np.ndarray:
        r"""The state's correlation matrix in the xxpp basis.

        Let :math:`M_{(xxpp)}` be the correlation matrix in the xxpp basis.
        Then

        .. math::
            M_{ij (xxpp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho,

        where :math:`M_{ij (xxpp)}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            Y = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray: The :math:`2d \times 2d` correlation matrix in the
            xxpp-basis.
        """
        np = self._connector.np

        xxpp_mean_vector = self.xxpp_mean_vector

        return self.xxpp_covariance_matrix + 2 * np.outer(
            xxpp_mean_vector, xxpp_mean_vector
        )

    @property
    def xxpp_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        The state's mean and correlation matrix ordered in the xxpp basis.

        Returns:
            tuple: :meth:`xxpp_mean_vector`, :meth:`xxpp_correlation_matrix`.
        """

        return self.xxpp_mean_vector, self.xxpp_correlation_matrix

    @property
    def xpxp_mean_vector(self) -> np.ndarray:
        r"""Returns the xpxp-ordered mean of the state.

        Returns:
            numpy.ndarray: A :math:`2d`-vector. The expectation value of the quadrature
            operators in `xxpp`-ordering, i.e. :math:`\operatorname{Tr} \rho R`, where
            :math:`R = (x_1, p_1, \dots, x_d, p_d)^T`.
        """
        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        return T @ self.xxpp_mean_vector

    @xpxp_mean_vector.setter
    def xpxp_mean_vector(self, value: np.ndarray) -> None:
        np = self._connector.np

        self._validate_mean(value, self.d)

        m = (value[::2] + 1j * value[1::2]) / np.sqrt(2 * self._config.hbar)

        self._m = m

    @property
    def xpxp_covariance_matrix(self) -> np.ndarray:
        r"""The `xpxp`-ordered coveriance matrix of the state.

        The `xpxp`-ordered covariance matrix :math:`\sigma` is defined by

        .. math::
            \sigma_{ij} := \langle R_i R_j + R_j R_i \rangle_\rho
                - 2 \langle R_i \rangle_\rho \langle R_j \rangle_\rho,

        where

        .. math::
            R = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray:
                The :math:`2d \times 2d` `xpxp`-ordered covariance matrix in
                xpxp-ordered basis.
        """

        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        return T @ self.xxpp_covariance_matrix @ T.transpose()

    @xpxp_covariance_matrix.setter
    def xpxp_covariance_matrix(self, new_cov: np.ndarray) -> None:
        np = self._connector.np

        d = self.d

        self._validate_cov(new_cov, d)

        T = from_xxpp_to_xpxp_transformation_matrix(d)

        dimensionless_cov = new_cov / self._config.hbar
        dimensionless_xp_cov = T.transpose() @ dimensionless_cov @ T

        blocks = (dimensionless_xp_cov - np.identity(2 * d)) / 4

        C_real = blocks[:d, :d] + blocks[d:, d:]
        G_real = blocks[:d, :d] - blocks[d:, d:]

        C_imag = blocks[:d, d:] - blocks[d:, :d]
        G_imag = blocks[:d, d:] + blocks[d:, :d]

        C = C_real + 1j * C_imag
        G = G_real + 1j * G_imag

        self._G = G
        self._C = C

    @property
    def xpxp_correlation_matrix(self) -> np.ndarray:
        r"""The `xpxp`-ordered correlation matrix of the state.

        Let :math:`M` be the correlation matrix in the `xpxp` basis.
        Then

        .. math::
            M_{ij} = \langle R_i R_j + R_j R_i \rangle_\rho,

        where :math:`M_{ij}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            R = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray: The :math:`2d \times 2d` `xpxp`-ordered correlation matrix.
        """

        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        return T @ self.xxpp_correlation_matrix @ T.transpose()

    @property
    def xpxp_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""The state's mean and correlation matrix ordered by the `xpxp` basis.

        Returns:
            tuple: :meth:`mean`, :meth:`corr`.
        """

        return self.xpxp_mean_vector, self.xpxp_correlation_matrix

    @property
    def complex_displacement(self) -> np.ndarray:
        r"""The complex displacement of the state.

        The complex displacement is defined by

        .. math::
            \mu_{c} := \begin{bmatrix}
                \langle a_1 \rangle_{\rho}, \dots, \langle a_d \rangle_{\rho},
                \langle a^\dagger_1 \rangle_{\rho},
                \dots, \langle a^\dagger_d \rangle_{\rho}
            \end{bmatrix}.

        Equivalently, one can write

        .. math::
            \mu_{c} := W \mu_{xxpp},

        where :math:`\mu_{xxpp}` is the xxpp-ordered mean vector
        :attr:`xxpp_mean_vector` and

        .. math::
            W = \frac{1}{\sqrt{2}} \begin{bmatrix}
                I_{d} & i I_{d} \\
                I_{d} & -i I_{d}
            \end{bmatrix}.

        Returns:
            numpy.ndarray: The complex displacement.
        """
        np = self._connector.np

        return np.concatenate([self._m, self._m.conj()])

    @property
    def complex_covariance(self) -> np.ndarray:
        r"""The complex covariance of the state.

        The complex covariance is defined by

        .. math::
            \sigma_{c, ij} = \langle
                \xi_i \xi^\dagger_j + \xi_j \xi^\dagger_i
            \rangle_{\rho}
            - 2 \langle \xi_i \rangle_{\rho} \langle \xi^\dagger_j \rangle_{\rho},

        where

        .. math::
            \xi = \begin{bmatrix}
                a_1, \dots a_d, a^\dagger_1, \dots, a^\dagger_d
            \end{bmatrix}.

        Equivalently, one can write

        .. math::
            \sigma_{c} = \frac{1}{\hbar} W \sigma_{xxpp} W^{\dagger},

        where :math:`\sigma_{xxpp}` is the xxpp-ordered covariance
        matrix :attr:`xxpp_cov` and

        .. math::
            W = \frac{1}{\sqrt{2}} \begin{bmatrix}
                I_{d} & i I_{d} \\
                I_{d} & -i I_{d}
            \end{bmatrix}.

        Returns:
            numpy.ndarray: The complex covariance.
        """

        np = self._connector.np

        return 2 * np.block(
            [[self._C.conj(), self._G], [self._G.conj(), self._C]]
        ) + np.identity(2 * self.d)

    @property
    def Q_matrix(self):
        return (self.complex_covariance + np.identity(2 * len(self))) / 2

    def rotated(self, phi: float) -> "GaussianState":
        r"""Returns a copy of the current Gaussian state, rotated by an angle `phi`.

        Let :math:`\phi \in [ 0, 2 \pi )`. Let us define the following:

        .. math::
            x_{i, \phi} = \cos\phi~x_i + \sin\phi~p_i,

        which is a generalized quadrature operator. One could rotate the state by a
        simple complex phase, which can be shown by using the transformation rules
        between the ladder operators and quadrature operators, i.e.

        .. math::
            x_i &= \sqrt{\frac{\hbar}{2}} (a_i + a_i^\dagger) \\
            p_i &= -i \sqrt{\frac{\hbar}{2}} (a_i - a_i^\dagger),

        which we could rewrite :math:`x_{i, \phi}` to

        .. math::
            x_{i, \phi} = \sqrt{\frac{\hbar}{2}} \left(
                a_i \exp(-i \phi) + a_i^\dagger \exp(i \phi)
            \right),

        meaning that e.g. the annihilation operators :math:`a_i` are transformed just
        by multiplying it with a phase factor :math:`\exp(-i \phi)` i.e.

        .. math::
            (\langle a_i \rangle_{\rho} =: )~m_i &\mapsto \exp(-i \phi) m_i \\
            (\langle a^\dagger_i a_j \rangle_{\rho} =: )~C_{ij} &\mapsto C_{ij} \\
            (\langle a_i a_j \rangle_{\rho} =: )~G_{ij} &\mapsto \exp(-i 2 \phi) G_{ij}.

        Args:
            phi (float): The angle to rotate the state with.

        Returns:
            GaussianState: The rotated `GaussianState` instance.
        """
        np = self._connector.np

        phase = np.exp(-1j * phi)

        return self.__class__._from_representation(
            C=self._C,
            G=(self._G * phase**2),
            m=(self._m * phase),
            config=self._config,
            connector=self._connector,
        )

    def reduced(self, modes: Tuple[int, ...]) -> "GaussianState":
        """Returns a copy of the current state, reduced to the given `modes`.

        This method essentially preserves the modes specified from the representation
        of the Gaussian state, but cuts out the other modes.

        Args:
            modes (tuple[int]): The modes to reduce the state to.

        Returns:
            GaussianState: The reduced `GaussianState` instance.
        """
        return self.__class__._from_representation(
            C=self._C[np.ix_(modes, modes)],
            G=self._G[np.ix_(modes, modes)],
            m=self._m[np.ix_(modes)],
            config=self._config,
            connector=self._connector,
        )

    def xpxp_reduced_rotated_mean_and_covariance(
        self, modes: Tuple[int, ...], phi: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""The mean and covariance on a rotated and reduced state.

        Let the index set :math:`\vec{i}` correspond to `modes`, and the angle
        :math:`\phi` correspond to `phi`. The current :class:`GaussianState` instance
        is reduced to `modes` and rotated by `phi` in a new instance, and let that
        state be denoted by :math:`\rho_{\vec{i}, \phi}`.

        Then the `xpxp`-ordered mean and covariance can be calculated by

        .. math::
            \mu_{\vec{i}, \phi}
                &:= \langle R_{\vec{i}} \rangle_{\rho_{\vec{i}, \phi}}, \\
            \sigma_{\vec{i}, \phi}
                &:=  \langle
                    R_{\vec{i}} R_{\vec{i}}^T
                \rangle_{\rho_{\vec{i}, \phi}}
                - \mu_{\vec{i}, \phi} \mu_{\vec{i}, \phi}^T,

        where

        .. math::
            R = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`R_{\vec{i}}` is just the same vector, reduced to a subsystem
        specified by :math:`\vec{i}`.

        Args:
            modes (tuple[int]): The modes to reduce the state to.
            phi (float): The angle to rotate the state with.

        Returns:
            (numpy.ndarray, numpy.ndarray):
                `xpxp`-ordered mean and covariance of the reduced and rotated
                version of the current :class:`GaussianState`.
        """
        transformed_state = self.reduced(modes).rotated(phi)

        return (
            transformed_state.xpxp_mean_vector,
            transformed_state.xpxp_covariance_matrix,
        )

    def mean_photon_number(self, modes: Optional[Tuple[int, ...]] = None) -> float:
        r"""This method returns the mean photon number of the given modes.
        The mean photon number :math:`\bar{n} = \langle \hat{n}
        \rangle` can be calculated in terms of the ladder operators by the following
        expression

        .. math::
            \sum_{i=1}^{d} \langle a_{i}^\dagger a_{i} \rangle =
            \operatorname{Tr}(\rho \hat{n}),

        where :math:`a`, :math:`a ^\dagger` are the annihilation and the creation
        operators respectively, :math:`\rho` is the density operator of the
        currently represented state and :math:`d` is the number of modes. For a general
        displaced squeezed gaussian state, the mean photon number is

        .. math::
            \langle \hat{n} \rangle = \operatorname{Tr}(\langle a^\dagger a \rangle) +
            \mu_{c}^ \dagger \cdot \mu_{c},

        where :math:`\mu_{c}` is the :attr:`complex_displacement`.

        .. note::
            This method can also be used to return the summation of the mean photon
            number for multiple modes if the mode parameter contains more than one
            integer e.g :math:`(0,1,...)`.

        Args:
            modes (tuple[int]): The correspoding modes at which the mean photon number
                is calculated.
        Returns:
            float: The expectation value of the photon number.
        """

        if modes is not None:
            return self.reduced(modes).mean_photon_number()

        np = self._connector.np

        return (np.trace(self._C) + self._m.conjugate() @ self._m).real

    def variance_photon_number(self, modes: Optional[Tuple[int, ...]] = None) -> float:
        r"""
        This method calculates the variance of the photon number operator as follows:

        .. math::
            \operatorname{Var}(\hat{n}) = \mathbb{E}((\hat{n} - \bar{n})^2),

        where :math:`\bar{n}` is the expectation value of the photon number, given by
        :meth:`mean_photon_number`.

        See, e.g., [Means and covariances of photon numbers in multimode Gaussian states](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.023817)
        for details on how to calculate this for Gaussian states.

        Args:
            modes (Tuple[int, ...]): The correspoding modes at which the variance of
                the photon number is calculated.

        Returns:
            float: Variance of the photon number operator
        """  # noqa: E501

        if modes is not None:
            return self.reduced(modes).variance_photon_number()

        np = self._connector.np

        d = self._d

        m = self._m

        m_outer = np.outer(m, m)

        G = self._G + m_outer
        C = self._C + np.outer(np.conj(m), m)

        correlation_term = (
            np.einsum("kj,jk", np.conj(G), G)
            + np.einsum("jk,kj", C, np.identity(d) + C)
            + np.einsum("jj,kk", C, C)
            - 2 * np.sum(np.abs(m_outer) ** 2)
        )

        mean = self.mean_photon_number()

        variance = (correlation_term - mean**2).real

        return variance

    def fidelity(self, state: "GaussianState") -> float:
        r"""Calculates the state fidelity between two quantum states.

        The state fidelity :math:`F` between two density matrices
        :math:`\rho_1, \rho_2` is given by:

        .. math::
            \operatorname{F}(\rho_1, \rho_2) = \left [
                \operatorname{Tr} \sqrt{\sqrt{\rho_1}
                \rho_2 \sqrt{\rho_1} }
            \right ]

        A Gaussian state can be represented by its covariance matrix and displacement
        vector.

        Let :math:`\mu_1, \mu_2` be the displacement vectors and
        :math:`\sigma_1, \sigma_2` be the covariance matrices of the
        :math:`\rho_1, \rho_2` Gaussian states, respectively. Define
        :math:`\hat{\sigma} = \frac{\sigma_1 + \sigma_2}{2}` and
        :math:`\Delta\mu = \mu_2 - \mu_1`.
        The fidelity can be written as

        .. math::
            \operatorname{F}(\rho_1, \rho_2) = \operatorname{F_0}(\sigma_1, \sigma_2)
            \exp \left(
                -\frac{1}{4} \Delta\mu^T (\hat{\sigma})^{-1} \Delta\mu
            \right),

        where :math:`F_0` can be written as

        .. math::
            \operatorname{F_0} = \frac{
                \prod_{i=1}^d \left [w_i + \sqrt{w_i^2 - 1} \right]
            }{
                \sqrt{\det \hat{\sigma}}
            },

        where :math:`w_i \geq 1` and :math:`\pm w_i` are the eigenvalues of the matrix

        .. math::
            W := - \frac{i}{2} \Omega^T \hat{\sigma}^{-1} \left(
                I - \sigma_2 \Omega \sigma_1 \Omega
            \right)

        and

        .. math::
            \Omega = \begin{bmatrix}
                0 & 1 \\-1 & 0
            \end{bmatrix} \otimes I_{2d \times 2d}.

        References:
            - `Quantum fidelity for arbitrary Gaussian states <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.260501>`_.

        Note:
            In this notation :math:`\sqrt{\hbar} \mu_i` is equivalent to
            :attr:`xpxp_mean_vector` and :math:`\hbar \sigma_i` is equivalent to
            :attr:`xpxp_covariance_matrix`.

        Args:
            state: Another :class:`~piquasso._simulators.gaussian.state.GaussianState`
                instance.

        Returns:
            float: The calculated fidelity.
        """  # noqa: E501

        np = self._connector.np

        hbar = self._config.hbar

        sigma_1 = self.xpxp_covariance_matrix / hbar
        sigma_2 = state.xpxp_covariance_matrix / hbar

        sigma_mean = (sigma_1 + sigma_2) / 2

        Omega = symplectic_form(self.d)

        Id = np.identity(2 * self.d)

        W_aux = (
            -1j
            / 2
            * Omega.T
            @ np.linalg.inv(sigma_mean)
            @ (Id - sigma_2 @ Omega @ sigma_1 @ Omega)
        )

        eigenvalues = np.linalg.eigvals(W_aux)
        positive_eigenvalues = eigenvalues[eigenvalues >= 0]

        F_0 = np.prod(
            [w + np.sqrt(w**2 - 1) for w in positive_eigenvalues]
        ) / np.sqrt(np.linalg.det(sigma_mean))

        mu_1 = self.xpxp_mean_vector / np.sqrt(hbar)
        mu_2 = state.xpxp_mean_vector / np.sqrt(hbar)
        delta_mu = mu_2 - mu_1
        displaced_contribition = np.exp(
            -1 / 2 * delta_mu @ np.linalg.inv(sigma_mean) @ delta_mu
        )

        return np.real(displaced_contribition * F_0)

    def quadratic_polynomial_expectation(
        self, A: np.ndarray, b: np.ndarray, c: float = 0.0, phi: float = 0.0
    ) -> float:
        r"""The expectation value of the specified quadratic polynomial.

        A quadratic polynomial can be written as

        .. math::
            f(R) = R^T A R + R \cdot b + c,

        where :math:`R = (x_1, p_1, \dots, x_d, p_d)^T` is the vector of the quadrature
        operators where :math:`d` is the number of modes,
        :math:`A \in \mathbb{R}^{2d \times 2d}` is a symmetric matrix,
        :math:`b \in \mathbb{R}^{2d}`, and :math:`c\in\mathbb{R}`.

        This method returns the expectation value :math:`E[f(R)]` using the following
        equation:

        .. math::
            \operatorname{E}[f(R)]
                = \operatorname{Tr}[ A\sigma ] + \mu^T A \mu + \mu^T b + c,

        where :math:`\sigma` is the covariance matrix, :math:`\mu = E[R]` is the mean
        of the quadrature operators and

        .. math::
            \operatorname{E}[\cdot] = \operatorname{Tr}[\cdot \rho],

        where :math:`\rho` is the density matrix of the state.

        Args:
            A (numpy.ndarray):
                A :math:`2d \times 2d` real symmetric matrix corresponding to the
                quadratic coefficients, where :math:`d` is the number of modes.
            b (numpy.ndarray):
                A one-dimensional :math:`2d`-length real-valued vector that corresponds
                to the first order terms of the quadratic polynomial.
            c (float): The constant term in the quadratic polynomial. Defaults to `0`.
            phi (float): Rotation angle, by which the state is rotated. Defaults to `0`.
        Returns:
            float: The expectation value of the quadratic polynomial.
        """

        np = self._connector.np

        if self._config.validate and not is_symmetric(A):
            raise InvalidParameter("The specified matrix is not symmetric.")

        state = self.rotated(phi)
        mean = state.xpxp_mean_vector
        cov = state.xpxp_covariance_matrix
        first_moment = np.trace(A @ cov) / 2 + mean @ A @ mean + mean @ b + c
        # TODO: calculate the variance.
        return first_moment

    def _is_displaced(self) -> bool:
        return not self._connector.np.allclose(self._m, 0.0)

    def _get_density_matrix_calculation(self) -> DensityMatrixCalculation:
        if self._connector.allow_conditionals and not self._is_displaced():
            return NondisplacedDensityMatrixCalculation(
                complex_covariance=self.complex_covariance, connector=self._connector
            )
        return DisplacedDensityMatrixCalculation(
            complex_displacement=self.complex_displacement,
            complex_covariance=self.complex_covariance,
            connector=self._connector,
        )

    @property
    def density_matrix(self) -> np.ndarray:
        calculation = self._get_density_matrix_calculation()

        return calculation.get_density_matrix(
            get_fock_space_basis(d=self.d, cutoff=self._config.cutoff)
        )

    def wigner_function(
        self,
        positions: List[List[float]],
        momentums: List[List[float]],
        modes: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        r"""
        This method calculates the Wigner function values at the specified position and
        momentum vectors, according to the following equation:

        .. math::
            W(r) = \frac{1}{\pi^d \sqrt{\mathrm{det} \sigma}}
                \exp \big (
                    - (r - \mu)^T
                    \sigma^{-1}
                    (r - \mu)
                \big ).

        Args:
            positions (list[list[float]]): List of position vectors.
            momentums (list[list[float]]): List of momentum vectors.
            modes (tuple[int], optional):
                Modes where Wigner function should be calculcated.

        Returns:
            numpy.ndarray:
                The Wigner function values in the shape of a grid specified by the
                input.
        """

        if modes:
            reduced_state = self.reduced(modes)
            return gaussian_wigner_function(
                positions,
                momentums,
                d=reduced_state.d,
                mean=reduced_state.xpxp_mean_vector,
                cov=reduced_state.xpxp_covariance_matrix,
            )

        return gaussian_wigner_function(
            positions,
            momentums,
            d=self.d,
            mean=self.xpxp_mean_vector,
            cov=self.xpxp_covariance_matrix,
        )

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        calculation = self._get_density_matrix_calculation()

        return self._connector.np.real(
            calculation.get_density_matrix_element(
                bra=occupation_number,
                ket=occupation_number,
            )
        )

    def get_threshold_detection_probability(
        self, occupation_number: Union[np.ndarray, Tuple[int, ...]]
    ) -> float:
        """Calculates the probability of threshold particle number detection events.

        The occupation number needs to be specified as consecutive 0s and 1s, where 0
        corresponds to no particle detection and 1 to detecting some (>0) particles.

        Args:
            occupation_number (Union[np.ndarray, Tuple[int, ...]]):
                Array of 0s and 1s corresponding to particle detection events per mode.

        Raises:
            PiquassoException: When an invalid input 'occupation_number' is specified.

        Returns:
            float: The threshold particle number detection probability.
        """

        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        occupation_number_np = np.array(occupation_number)

        if self._config.validate and not np.array_equal(
            occupation_number_np, occupation_number_np.astype(bool)
        ):
            raise PiquassoException(
                f"The specified occupation numbers must only contain '0' or '1': "
                f"occupation_number='{occupation_number}'."
            )

        hbar = self._config.hbar

        if not self._is_displaced():
            return calculate_click_probability_nondisplaced(
                self.xpxp_covariance_matrix / hbar,
                tuple(occupation_number),
            )

        return calculate_click_probability(
            self.xpxp_covariance_matrix / hbar,
            self.xpxp_mean_vector / np.sqrt(hbar),
            tuple(occupation_number),
        )

    @property
    def fock_probabilities(self) -> np.ndarray:
        calculation = self._get_density_matrix_calculation()

        return calculation.get_particle_number_detection_probabilities(
            get_fock_space_basis(d=self.d, cutoff=self._config.cutoff)
        )

    def is_pure(self) -> bool:
        d = self.d
        cov = self.xpxp_covariance_matrix / self._config.hbar

        J = np.array([[0, 1], [-1, 0]], dtype=cov.dtype)
        Jp = block_diag(*[J] * self.d)

        return np.allclose(np.linalg.matrix_power(Jp @ cov, 2), -np.identity(2 * d))

    def purify(self) -> "GaussianState":
        """
        Returns a purification corresponding to the current state on a double-sized
        system.

        Note, that the purification results in a state where the mean vector on the
        auxiliary system is identical with the mean vector in the original system.

        Source:
            - `Evaluating capacities of Bosonic Gaussian channels <https://arxiv.org/abs/quant-ph/9912067>`_

        Returns:
            GaussianState: The purified Gaussian state.
        """  # noqa: E501
        np = self._connector.np

        hbar = self._config.hbar
        cov = self.xpxp_covariance_matrix / hbar

        d = self.d

        T = from_xxpp_to_xpxp_transformation_matrix(d)

        S, D = williamson(T.T @ cov @ T, self._connector)

        beta_diagonals = np.diag(np.sqrt(np.abs(np.diag(D)[:d] ** 2 - 1)))
        zeros = np.zeros_like(beta_diagonals)

        beta_skeleton = np.block([[zeros, beta_diagonals], [beta_diagonals, zeros]])

        TS = T @ S

        beta = TS @ beta_skeleton @ TS.T

        purified_cov = np.block(
            [
                [cov, beta],
                [beta, cov],
            ]
        )

        purification = GaussianState(
            d=2 * d, connector=self._connector, config=self._config
        )

        purification.xpxp_covariance_matrix = hbar * purified_cov

        mean = self.xpxp_mean_vector
        purification.xpxp_mean_vector = np.concatenate([mean] * 2)

        return purification
