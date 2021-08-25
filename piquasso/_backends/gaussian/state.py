#
# Copyright 2021 Budapest Quantum Computing Group
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
import random
from functools import lru_cache
from itertools import repeat
from typing import Tuple, List, Callable

import numpy as np
import numpy.typing as npt
import scipy

from piquasso._math._random import choose_from_cumulated_probabilities
from piquasso._math.combinatorics import get_occupation_numbers
from piquasso._math.functions import gaussian_wigner_function
from piquasso._math.linalg import (
    is_symmetric,
    symplectic_form,
    is_positive_semidefinite,
)
from piquasso.api import constants
from piquasso.api.errors import InvalidState, InvalidParameter
from piquasso.api.state import State
from .circuit import GaussianCircuit
from .probabilities import (
    DensityMatrixCalculation,
    ThresholdCalculation,
)
from .transformations import quad_transformation


class GaussianState(State):
    r"""Class to represent a Gaussian state.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=5) | pq.Vacuum()

    """
    _m: npt.NDArray[np.complex128]
    _G: npt.NDArray[np.complex128]
    _C: npt.NDArray[np.complex128]

    circuit_class = GaussianCircuit

    def __init__(self, d: int) -> None:
        self._d = d
        self.reset()

    @property
    def d(self) -> int:
        return self._d

    def reset(self) -> None:
        """
        Resets this object to a vacuum state.
        """

        vector_shape = (self.d, )
        matrix_shape = vector_shape * 2

        self._m = np.zeros(vector_shape, dtype=complex)
        self._G = np.zeros(matrix_shape, dtype=complex)
        self._C = np.zeros(matrix_shape, dtype=complex)

    @classmethod
    def _from_representation(
        cls, *,
        m: npt.NDArray[np.complex128],
        G: npt.NDArray[np.complex128],
        C: npt.NDArray[np.complex128],
    ) -> "GaussianState":
        obj = cls(d=len(m))

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
        """
        Validates the Gaussian state.

        Raises:
            InvalidState:
                Raised if the underlying Gaussian state is invalid, which could mean
                - ill-shaped mean and covariance;
                - non-symmetric covariance matrix;
                - the covariance matrix doesn't fulfill the Robertson-Schrödinger
                  uncertainty relations.
        """

        self._validate_mean(self.mean, self.d)
        self._validate_cov(self.cov, self.d)

    @staticmethod
    def _validate_mean(mean: npt.NDArray[np.float64], d: int) -> None:
        expected_shape = (2 * d, )

        if not mean.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'mean' vector shape; "
                f"expected={expected_shape}, actual={mean.shape}."
            )

    @staticmethod
    def _validate_cov(cov: npt.NDArray[np.float64], d: int) -> None:
        expected_shape = (2 * d, ) * 2

        if not cov.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'cov' matrix shape; "
                f"expected={expected_shape}, actual={cov.shape}."
            )

        if not is_symmetric(cov):
            raise InvalidState("The covariance matrix is not symmetric.")

        if not is_positive_semidefinite(cov / constants.HBAR + 1j * symplectic_form(d)):
            raise InvalidState(
                "The covariance matrix is invalid, since it doesn't fulfill the "
                "Robertson-Schrödinger uncertainty relation."
            )

    @property
    def xp_mean(self) -> npt.NDArray[np.float64]:
        r"""The state's mean in the xp-ordered basis.

        The expectation value of the quadrature operators in xp basis, i.e.
        :math:`\operatorname{Tr} \rho Y`, where
        :math:`Y = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.

        Returns:
            numpy.ndarray: A :math:`d`-vector.
        """

        dimensionless_xp_mean = np.concatenate(
            [self._m.real, self._m.imag]
        ) * np.sqrt(2)

        return dimensionless_xp_mean * np.sqrt(constants.HBAR)

    @property
    def xp_cov(self) -> npt.NDArray[np.float64]:
        r"""The xp-ordered coveriance matrix of the state.

        The xp-ordered covariance matrix :math:`\sigma_{xp}` is defined by

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

        C = self._C
        G = self._G

        dimensionless_xp_cov = 2 * np.block(
            [
                [(G + C).real, (G + C).imag],
                [(G - C).imag, (-G + C).real],
            ],
        ) + np.identity(2 * self.d)

        return dimensionless_xp_cov * constants.HBAR

    @property
    def xp_corr(self) -> npt.NDArray[np.float64]:
        r"""The state's correlation matrix in the xp basis.

        Let :math:`M_{(xp)}` be the correlation matrix in the xp basis.
        Then

        .. math::
            M_{ij (xp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho,

        where :math:`M_{ij (xp)}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            Y = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray: The :math:`2d \times 2d` correlation matrix in the xp basis.
        """
        xp_mean = self.xp_mean
        return self.xp_cov + 2 * np.outer(xp_mean, xp_mean)

    @property
    def xp_representation(self) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        r"""
        The state's mean and correlation matrix ordered in the xp basis.

        Returns:
            tuple: :meth:`xp_mean`, :meth:`xp_corr`.
        """

        return self.xp_mean, self.xp_corr

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        r"""Returns the xp-ordered mean of the state.

        Returns:
            numpy.ndarray: A :math:`2d`-vector.
                The expectation value of the quadrature operators in
                xp-ordering, i.e. :math:`\operatorname{Tr} \rho R`, where
                :math:`R = (x_1, p_1, \dots, x_d, p_d)^T`.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_mean

    @mean.setter
    def mean(self, new_mean: npt.NDArray[np.float64]) -> None:
        self._validate_mean(new_mean, self.d)

        m = (new_mean[::2] + 1j * new_mean[1::2]) / np.sqrt(2 * constants.HBAR)

        self._m = m

    @property
    def cov(self) -> npt.NDArray[np.float64]:
        r"""The quadrature-ordered coveriance matrix of the state.

        The quadrature-ordered covariance matrix :math:`\sigma` is defined by

        .. math::
            \sigma_{ij} := \langle R_i R_j + R_j R_i \rangle_\rho
                - 2 \langle R_i \rangle_\rho \langle R_j \rangle_\rho,

        where

        .. math::
            R = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray:
                The :math:`2d \times 2d` quadrature-ordered covariance matrix in
                xp-ordered basis.
        """

        T = quad_transformation(self.d)
        return T @ self.xp_cov @ T.transpose()

    @cov.setter
    def cov(self, new_cov: npt.NDArray[np.float64]) -> None:
        d = self.d

        self._validate_cov(new_cov, d)

        T = quad_transformation(d)

        dimensionless_cov = new_cov / constants.HBAR
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
    def corr(self) -> npt.NDArray[np.float64]:
        r"""The quadrature-ordered correlation matrix of the state.

        Let :math:`M` be the correlation matrix in the quadrature basis.
        Then

        .. math::
            M_{ij} = \langle R_i R_j + R_j R_i \rangle_\rho,

        where :math:`M_{ij}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            R = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            numpy.ndarray: The :math:`2d \times 2d` quad-ordered correlation matrix.
        """

        T = quad_transformation(self.d)
        return T @ self.xp_corr @ T.transpose()

    @property
    def quad_representation(self) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        r"""The state's mean and correlation matrix ordered by the quadrature basis.

        Returns:
            tuple: :meth:`mean`, :meth:`corr`.
        """

        return self.mean, self.corr

    @property
    def complex_displacement(self) -> npt.NDArray[np.complex128]:
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
            \mu_{c} := W \mu_{xp},

        where :math:`\mu_{xp}` is the xp-ordered mean vector :attr:`xp_mean`
        and

        .. math::
            W = \frac{1}{\sqrt{2}} \begin{bmatrix}
                I_{d} & i I_{d} \\
                I_{d} & -i I_{d}
            \end{bmatrix}.

        Returns:
            numpy.ndarray: The complex displacement.
        """

        return np.concatenate([self._m, self._m.conj()])

    @property
    def complex_covariance(self) -> npt.NDArray[np.complex128]:
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
            \sigma_{c} = \frac{1}{\hbar} W \sigma_{xp} W^{\dagger},

        where :math:`\sigma_{xp}` is the xp-ordered covariance matrix :attr:`xp_cov`
        and

        .. math::
            W = \frac{1}{\sqrt{2}} \begin{bmatrix}
                I_{d} & i I_{d} \\
                I_{d} & -i I_{d}
            \end{bmatrix}.

        Returns:
            numpy.ndarray: The complex covariance.
        """

        return 2 * np.block(
            [
                [self._C.conj(), self._G],
                [self._G.conj(), self._C]
            ]
        ) + np.identity(2 * self.d)

    def rotated(self, phi: float) -> "GaussianState":
        r"""Returns the copy of the current state, rotated by angle `phi`.

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

        meaning that e.g. the annihilation operators `a_i` are transformed just
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

        return GaussianState._from_representation(
            C=self._C,
            G=(self._G * phase**2),
            m=(self._m * phase),
        )

    def reduced(self, modes: Tuple[int, ...]) -> "GaussianState":
        """Returns the copy of the current state, reduced to the given `modes`.

        This method essentially preserves the modes specified from the representation
        of the Gaussian state, but cuts out the other modes.

        Args:
            modes (tuple[int]): The modes to reduce the state to.

        Returns:
            GaussianState: The reduced `GaussianState` instance.
        """
        return GaussianState._from_representation(
            C=self._C[np.ix_(modes, modes)],
            G=self._G[np.ix_(modes, modes)],
            m=self._m[np.ix_(modes)],
        )

    def reduced_rotated_mean_and_cov(
        self, modes: Tuple[int, ...], phi: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""The mean and covariance on a rotated and reduced state.

        Let the index set :math:`\vec{i}` correspond to `modes`, and the angle
        :math:`\phi` correspond to `phi`. The current :class:`GaussianState` instance
        is reduced to `modes` and rotated by `phi` in a new instance, and let that
        state be denoted by :math:`\rho_{\vec{i}, \phi}`.

        Then the quadrature ordered mean and covariance can be calculated by

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
                Quadrature ordered mean and covariance of the reduced and rotated
                version of the current :class:`GaussianState`.
        """
        transformed_state = self.reduced(modes).rotated(phi)

        return transformed_state.mean, transformed_state.cov

    def mean_photon_number(self, modes: Tuple[int, ...]) -> float:
        r""" This method returns the mean photon number of the given modes.
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

        state = self.reduced(modes)
        return (np.trace(state._C) + state._m.conjugate() @ state._m).real

    def quadratic_polynomial_expectation(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: float = 0.0,
        phi: float = 0.0
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

        if not is_symmetric(A):
            raise InvalidParameter("The specified matrix is not symmetric.")

        state = self.rotated(phi)
        mean = state.mean
        cov = state.cov
        first_moment = np.trace(A @ cov) / 2 + mean @ A @ mean + mean @ b + c
        # TODO: calculate the variance.
        return first_moment

    def get_density_matrix(self, cutoff: int) -> npt.NDArray[np.complex128]:
        calculation = DensityMatrixCalculation(
            complex_displacement=self.complex_displacement,
            complex_covariance=self.complex_covariance,
        )

        return calculation.get_density_matrix(
            get_occupation_numbers(d=self.d, cutoff=cutoff)
        )

    def wigner_function(
        self,
        positions: List[List[float]],
        momentums: List[List[float]],
        modes: Tuple[int, ...] = None
    ) -> npt.NDArray[np.float64]:
        r"""
        Calculates the Wigner function values at the specified position and momentum
        vectors, according to the equation

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
                mean=reduced_state.mean,
                cov=reduced_state.cov
            )

        return gaussian_wigner_function(
            positions,
            momentums,
            d=self.d,
            mean=self.mean,
            cov=self.cov,
        )

    def _apply_passive_linear(
        self, T: npt.NDArray[np.complex128], modes: Tuple[int, ...]
    ) -> None:
        self._m[modes, ] = T @ self._m[modes, ]

        self._apply_passive_linear_to_C_and_G(T, modes=modes)

    def _apply_passive_linear_to_C_and_G(
        self, T: npt.NDArray[np.complex128], modes: Tuple[int, ...]
    ) -> None:
        index = self._get_operator_index(modes)

        self._C[index] = T.conjugate() @ self._C[index] @ T.transpose()
        self._G[index] = T @ self._G[index] @ T.transpose()

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if len(auxiliary_modes) != 0:
            self._apply_passive_linear_to_auxiliary_modes(T, modes, auxiliary_modes)

    def _apply_passive_linear_to_auxiliary_modes(
        self,
        T: npt.NDArray[np.complex128],
        modes: Tuple[int, ...],
        auxiliary_modes: Tuple[int, ...],
    ) -> None:
        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        self._C[auxiliary_index] = T.conjugate() @ self._C[auxiliary_index]
        self._G[auxiliary_index] = T @ self._G[auxiliary_index]

        self._C[:, modes] = np.conj(self._C[modes, :]).transpose()
        self._G[:, modes] = self._G[modes, :].transpose()

    def _apply_linear(
        self,
        passive_block: npt.NDArray[np.complex128],
        active_block: npt.NDArray[np.complex128],
        modes: Tuple[int, ...],
    ) -> None:
        self._m[modes, ] = (
            passive_block @ self._m[modes, ]
            + active_block @ np.conj(self._m[modes, ])
        )

        self._apply_linear_to_C_and_G(passive_block, active_block, modes)

    def _apply_linear_to_C_and_G(
        self,
        P: npt.NDArray[np.complex128],
        A: npt.NDArray[np.complex128],
        modes: Tuple[int, ...]
    ) -> None:
        index = self._get_operator_index(modes)

        original_C = self._C[index]
        original_G = self._G[index]

        self._G[index] = (
            P @ original_G @ P.transpose()
            + A @ original_G.conjugate().transpose() @ A.transpose()
            + P @ (original_C.transpose() + np.identity(len(modes))) @ A.transpose()
            + A @ original_C @ P.transpose()
        )

        self._C[index] = (
            P.conjugate() @ original_C @ P.transpose()
            + A.conjugate() @ (
                original_C.transpose() + np.identity(len(modes))
            ) @ A.transpose()
            + P.conjugate() @ original_G.conjugate().transpose() @ A.transpose()
            + A.conjugate() @ original_G @ P.transpose()
        )

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if len(auxiliary_modes) != 0:
            self._apply_linear_to_auxiliary_modes(P, A, modes, auxiliary_modes)

    def _apply_linear_to_auxiliary_modes(
        self,
        P: npt.NDArray[np.complex128],
        A: npt.NDArray[np.complex128],
        modes: Tuple[int, ...],
        auxiliary_modes: Tuple[int, ...],
    ) -> None:
        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        auxiliary_C = self._C[auxiliary_index]
        auxiliary_G = self._G[auxiliary_index]

        self._C[auxiliary_index] = (
            P.conjugate() @ auxiliary_C
            + A.conjugate() @ auxiliary_G
        )

        self._G[auxiliary_index] = (
            P @ auxiliary_G
            + A @ auxiliary_C
        )

        self._C[:, modes] = self._C[modes, :].conjugate().transpose()
        self._G[:, modes] = self._G[modes, :].transpose()

    def _apply_displacement(
        self, displacement_vector: npt.NDArray[np.complex128], modes: Tuple[int, ...]
    ) -> None:
        self._m[modes, ] += displacement_vector

    def _apply_generaldyne_measurement(
        self, *,
        detection_covariance: npt.NDArray[np.float64],
        modes: Tuple[int, ...],
        shots: int
    ) -> npt.NDArray[np.float64]:
        d = self.d

        indices = []

        for mode in modes:
            indices.extend([2 * mode, 2 * mode + 1])

        outer_indices = np.delete(np.arange(2 * d), indices)

        r = self.mean

        r_measured = r[indices]
        r_outer = r[outer_indices]

        rho = self.cov

        rho_measured = rho[np.ix_(indices, indices)]
        rho_outer = rho[np.ix_(outer_indices, outer_indices)]
        rho_correlation = rho[np.ix_(outer_indices, indices)]

        rho_m = constants.HBAR * scipy.linalg.block_diag(
            *[detection_covariance] * len(modes)
        )

        # HACK: We need tol=1e-7 to avoid Numpy warnings at homodyne detection with
        # squeezed detection covariance. Numpy warns
        # 'covariance is not positive-semidefinite.', but it definitely is. In the SVG
        # decomposition (which numpy uses for the multivariate normal distribution)
        # the U^T and V matrices should equal, but our covariance might contain too
        # large values leading to inequality, resulting in warning.
        #
        # We might be better of setting `check_valid='ignore'` and verifying
        # postive-definiteness for ourselves.
        samples = np.random.multivariate_normal(
            mean=r_measured,
            cov=(rho_measured + rho_m),
            size=shots,
            tol=1e-7,
        )

        # NOTE: We choose the last sample for multiple shots.
        sample = samples[-1]

        evolved_rho_outer = (
            rho_outer
            - rho_correlation
            @ np.linalg.inv(rho_measured + rho_m)
            @ rho_correlation.transpose()
        )

        evolved_r_A = (
            r_outer
            + rho_correlation
            @ np.linalg.inv(rho_measured + rho_m)
            @ (sample - r_measured)
        )

        evolved_mean = np.zeros(shape=(2 * d, ))
        evolved_mean[outer_indices] = evolved_r_A

        evolved_cov: npt.NDArray[np.float64] = \
            np.identity(2 * d) * constants.HBAR
        evolved_cov[np.ix_(outer_indices, outer_indices)] = evolved_rho_outer

        self.mean = evolved_mean
        self.cov = evolved_cov

        return samples

    def _apply_particle_number_measurement(
        self,
        *,
        cutoff: int,
        modes: Tuple[int, ...],
        shots: int,
    ) -> List[List[int]]:
        def calculate_particle_number_detection_probability(
            state: "GaussianState",
            occupation_numbers: Tuple[int, ...],
        ) -> float:
            calculation = DensityMatrixCalculation(
                state.complex_displacement,
                state.complex_covariance,
            )

            return calculation.get_density_matrix_element(
                bra=occupation_numbers,
                ket=occupation_numbers,
            )

        return self._perform_sampling(
            cutoff=cutoff,
            modes=modes,
            shots=shots,
            calculation=calculate_particle_number_detection_probability,
        )

    def _apply_threshold_measurement(
        self,
        *,
        shots: int,
        modes: Tuple[int, ...],
    ) -> List[List[int]]:
        """
        NOTE: The same logic is used here, as for the particle number measurement.
        However, at threshold measurement there is no sense of cutoff, therefore it is
        set to 2 to make the logic sensible in this case as well.

        Also note, that one could speed up this calculation by not calculating the
        probability of clicks (i.e. 1 as sample), and argue that the probability of a
        click is equal to one minus the probability of no click.
        """
        if not np.allclose(self.mean, np.zeros_like(self.mean)):
            raise NotImplementedError(
                "Threshold measurement for displaced states are not supported: "
                f"mean={self.mean}"
            )

        def calculate_threshold_detection_probability(
            state: "GaussianState",
            occupation_numbers: Tuple[int, ...],
        ) -> float:
            calculation = ThresholdCalculation(state.xp_cov)

            return calculation.calculate_click_probability(occupation_numbers)

        return self._perform_sampling(
            cutoff=2,
            modes=modes,
            shots=shots,
            calculation=calculate_threshold_detection_probability,
        )

    def _perform_sampling(
        self,
        *,
        cutoff: int,
        modes: Tuple[int, ...],
        shots: int,
        calculation: Callable[["GaussianState", Tuple[int, ...]], float],
    ) -> List[List[int]]:
        @lru_cache(constants.cache_size)
        def get_probability(
            *, subspace_modes: Tuple[int, ...], occupation_numbers: Tuple[int, ...]
        ) -> float:
            reduced_state = self.reduced(subspace_modes)
            probability = calculation(
                reduced_state,
                occupation_numbers,
            )

            return max(probability, 0.0)

        samples = []

        for _ in repeat(None, shots):
            sample: List[int] = []

            previous_probability = 1.0

            for k in range(1, len(modes) + 1):
                subspace_modes = tuple(modes[:k])

                cumulated_probabilities = [0.0]

                guess = random.uniform(0.0, 1.0)

                choice = None

                for n in range(cutoff):
                    occupation_numbers = tuple(sample + [n])

                    probability = get_probability(
                        subspace_modes=subspace_modes,
                        occupation_numbers=occupation_numbers
                    )
                    conditional_probability = probability / previous_probability
                    cumulated_probabilities.append(
                        conditional_probability + cumulated_probabilities[-1]
                    )
                    if guess < cumulated_probabilities[-1]:
                        choice = n
                        break

                else:
                    choice = choose_from_cumulated_probabilities(
                        cumulated_probabilities
                    )

                previous_probability = (
                    cumulated_probabilities[choice + 1]
                    - cumulated_probabilities[choice]
                ) * previous_probability

                sample.append(choice)

            samples.append(sample)

        return samples

    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        calculation = DensityMatrixCalculation(
            self.complex_displacement,
            self.complex_covariance,
        )

        return calculation.get_density_matrix_element(
            bra=occupation_number,
            ket=occupation_number,
        )

    def get_fock_probabilities(self, cutoff: int) -> npt.NDArray[np.float64]:
        calculation = DensityMatrixCalculation(
            complex_displacement=self.complex_displacement,
            complex_covariance=self.complex_covariance,
        )

        return calculation.get_particle_number_detection_probabilities(
            get_occupation_numbers(d=self.d, cutoff=cutoff)
        )
