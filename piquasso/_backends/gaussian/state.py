#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import scipy
from scipy.special import factorial

import random

from itertools import repeat

from piquasso.api.state import State
from piquasso.api import constants
from piquasso.api.errors import InvalidState
from piquasso._math.functions import gaussian_wigner_function
from piquasso._math.hafnian import hafnian
from piquasso._math.linalg import (
    is_symmetric,
    symplectic_form,
    is_positive_semidefinite,
    block_reduce,
)

from .circuit import GaussianCircuit

from .transformations import quad_transformation


class GaussianState(State):
    r"""Object to represent a Gaussian state.

    A Gaussian state is fully characterised by its m and correlation
    matrix, i.e. its first and second moments with the quadrature
    operators.

    However, for computations, we only use the
    :math:`C, G \in \mathbb{C}^{d \times d}`
    and the :math:`m \in \mathbb{C}^d` vector.

    Attributes:
        m (numpy.array): The expectation value of the annihilation operators on all
            modes (a vector, essentially), and is defined by

            .. math::
                m = \langle \hat{a}_i \rangle_{\rho}.

        C (numpy.array): A correlation matrix which is defined by

            .. math::
                \langle \hat{C}_{ij} \rangle_{\rho} =
                \langle \hat{a}^\dagger_i \hat{a}_j \rangle_{\rho}
                - \langle \hat{a}^\dagger_i \rangle_{\rho}
                \langle \hat{a}_j \rangle_{\rho}.

        G (numpy.array): A correlation matrix which is defined by

            .. math::
                \langle \hat{G}_{ij} \rangle_{\rho} =
                \langle \hat{a}_i \hat{a}_j \rangle_{\rho}
                - \langle \hat{a}_i \rangle_{\rho} \langle \hat{a}_j \rangle_{\rho}.
    """

    circuit_class = GaussianCircuit

    def __init__(self, *, d):
        self.d = d
        self.reset()

    def reset(self):
        vector_shape = (self.d, )
        matrix_shape = vector_shape * 2

        self._m = np.zeros(vector_shape, dtype=complex)
        self._G = np.zeros(matrix_shape, dtype=complex)
        self._C = np.zeros(matrix_shape, dtype=complex)

    @classmethod
    def _from_representation(cls, *, m, G, C):
        obj = cls(d=len(m))

        obj._m = m
        obj._G = G
        obj._C = C

        return obj

    def __eq__(self, other):
        return (
            np.allclose(self._C, other._C)
            and np.allclose(self._G, other._G)
            and np.allclose(self._m, other._m)
        )

    def validate(self):
        self._validate_mean(self.mean, self.d)
        self._validate_cov(self.cov, self.d)

    @staticmethod
    def _validate_mean(mean, d):
        expected_shape = (2 * d, )

        if not mean.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'mean' vector shape; "
                f"expected={expected_shape}, actual={mean.shape}."
            )

    @staticmethod
    def _validate_cov(cov, d):
        expected_shape = (2 * d, ) * 2

        if not cov.shape == expected_shape:
            raise InvalidState(
                f"Invalid 'cov' matrix shape; "
                f"expected={expected_shape}, actual={cov.shape}."
            )

        if not is_symmetric(cov):
            raise InvalidState("The covariance matrix is not symmetric.")

        if not is_positive_semidefinite(cov + 1j * symplectic_form(d)):
            raise InvalidState(
                "The covariance matrix is invalid. TODO: We should have a link here."
            )

    @property
    def xp_mean(self):
        r"""The state's mean in the xp basis.

        The expectation value of the quadrature operators in xp basis, i.e.
        :math:`\operatorname{Tr} \rho \hat{Y}`, where
        :math:`\hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.

        Returns:
            np.array: A :math:`d`-vector.
        """

        _xp_mean = np.empty(2 * self.d)
        _xp_mean[:self.d] = self._m.real * np.sqrt(2 * constants.HBAR)
        _xp_mean[self.d:] = self._m.imag * np.sqrt(2 * constants.HBAR)
        return _xp_mean

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

        C = self._C
        G = self._G

        corr = 2 * np.block(
            [
                [(G + C).real, (G + C).imag],
                [(G - C).imag, (-G + C).real],
            ]
        ) + np.identity(2 * self.d)

        return corr * constants.HBAR

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
        xp_mean = self.xp_mean
        return self.xp_cov + 2 * np.outer(xp_mean, xp_mean)

    @property
    def xp_representation(self):
        r"""
        The state's mean and correlation matrix ordered in the xp basis.

        Returns:
            tuple: :meth:`xp_mean`, :meth:`xp_corr`.
        """

        return self.xp_mean, self.xp_corr

    @property
    def mean(self):
        r"""Returns the xp-ordered mean of the state.

        Returns:
            np.array: A :math:`2d`-vector.
                The expectation value of the quadrature operators in
                xp-ordering, i.e. :math:`\operatorname{Tr} \rho \hat{R}`, where
                :math:`\hat{R} = (x_1, p_1, \dots, x_d, p_d)^T`.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_mean

    @mean.setter
    def mean(self, new_mean):
        self._validate_mean(new_mean, self.d)

        m = (new_mean[::2] + 1j * new_mean[1::2]) / np.sqrt(2 * constants.HBAR)

        self._m = m

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
            np.array:
                The :math:`2d \times 2d` quadrature-ordered covariance matrix in
                xp-ordered basis.
        """

        T = quad_transformation(self.d)
        return T @ self.xp_cov @ T.transpose()

    @cov.setter
    def cov(self, new_cov):
        d = self.d

        self._validate_cov(new_cov, d)

        T = quad_transformation(d)

        xp_cov = (
            (T.transpose() @ new_cov @ T) / constants.HBAR
        )

        xp_cov = (xp_cov - np.identity(2 * d)) / 2

        C_real = (xp_cov[:d, :d] + xp_cov[d:, d:]) / 2
        G_real = (xp_cov[:d, :d] - xp_cov[d:, d:]) / 2

        C_imag = (xp_cov[:d, d:] - xp_cov[d:, :d]) / 2
        G_imag = (xp_cov[:d, d:] + xp_cov[d:, :d]) / 2

        C = C_real + 1j * C_imag
        G = G_real + 1j * G_imag

        self._G = G
        self._C = C

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
    def quad_representation(self):
        r"""The state's mean and correlation matrix ordered by the quadrature basis.

        Returns:
            tuple: :meth:`mean`, :meth:`corr`.
        """

        return self.mean, self.corr

    @property
    def husimi(self):
        return self.cov + np.identity(2 * self.d, dtype=complex)

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

        return GaussianState._from_representation(
            C=self._C,
            G=(self._G * phase**2),
            m=(self._m * phase),
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
        return GaussianState._from_representation(
            C=self._C[np.ix_(modes, modes)],
            G=self._G[np.ix_(modes, modes)],
            m=self._m[np.ix_(modes)],
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

        return transformed_state.mean, transformed_state.cov

    def wigner_function(self, quadrature_matrix, modes=None):
        r"""
        Calculates the Wigner function values at the specified `quadrature_matrix`,
        according to the equation

        .. math::
            W(r) = \frac{1}{\pi^d \sqrt{\mathrm{det} \sigma}}
                \exp \big (
                    - (r - \mu)^T
                    \sigma^{-1}
                    (r - \mu)
                \big ).

        Args:
            quadrature_matrix (list): list of canonical coordinates vectors.
            modes (tuple or None): modes where Wigner function should be calculcated.

        Returns:
            tuple: The Wigner function values in the shape of `quadrature_matrix`.
        """

        if modes:
            reduced_state = self.reduced(modes)
            return gaussian_wigner_function(
                quadrature_matrix,
                d=reduced_state.d,
                mean=reduced_state.mean,
                cov=reduced_state.cov
            )

        return gaussian_wigner_function(
            quadrature_matrix,
            d=self.d,
            mean=self.mean,
            cov=self.cov,
        )

    def _apply_passive_linear(self, T, modes):
        r"""Applies the passive transformation `T` to the quantum state.

        See:
            :ref:`passive_gaussian_transformations`

        Args:
            T (numpy.array): The matrix to be applied.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self._m[modes, ] = T @ self._m[modes, ]

        self._apply_passive_linear_to_C_and_G(T, modes=modes)

    def _apply_passive_linear_to_C_and_G(self, T, modes):
        index = self._get_operator_index(modes)

        self._C[index] = T.conjugate() @ self._C[index] @ T.transpose()
        self._G[index] = T @ self._G[index] @ T.transpose()

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if auxiliary_modes.size != 0:
            self._apply_passive_linear_to_auxiliary_modes(T, modes, auxiliary_modes)

    def _apply_passive_linear_to_auxiliary_modes(self, T, modes, auxiliary_modes):
        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        self._C[auxiliary_index] = T.conjugate() @ self._C[auxiliary_index]
        self._G[auxiliary_index] = T @ self._G[auxiliary_index]

        self._C[:, modes] = np.conj(self._C[modes, :]).transpose()
        self._G[:, modes] = self._G[modes, :].transpose()

    def _apply_linear(self, P, A, modes):
        r"""Applies an active transformation to the quantum state.

        See:
            :ref:`active_gaussian_transformations`

        Args:
            P (np.array): A matrix that represents a (P)assive transformation.
            A (np.array): A matrix that represents an (A)ctive transformation.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self._m[modes, ] = (
            P @ self._m[modes, ]
            + A @ np.conj(self._m[modes, ])
        )

        self._apply_linear_to_C_and_G(P, A, modes)

    def _apply_linear_to_C_and_G(self, P, A, modes):
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

        if auxiliary_modes.size != 0:
            self._apply_linear_to_auxiliary_modes(P, A, modes, auxiliary_modes)

    def _apply_linear_to_auxiliary_modes(self, P, A, modes, auxiliary_modes):
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

    def _apply_displacement(self, alpha, mode):
        self._m[mode] += alpha

    def _apply_generaldyne_measurement(self, *, detection_covariance, modes, shots):
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

        rho_m = scipy.linalg.block_diag(*[detection_covariance] * len(modes))

        # HACK: We need tol=1e-7 to avoid Numpy warnings at homodyne detection with
        # squeezed detection covariance. Numpy warns
        # 'covariance is not positive-semidefinite.', but it definitely is. In the SVG
        # decomposition (which numpy uses for the multivariate normal distribution)
        # the U^T and V matrices should equal, but our covariance might contain too
        # large values leading to inequality, resulting in warning.
        #
        # We might be better of setting `check_valid='ignore'` and verifying
        # postive-definiteness for ourselves.
        outcomes = np.random.multivariate_normal(
            mean=r_measured,
            cov=(rho_measured + rho_m),
            size=shots,
            tol=1e-7,
        )

        # NOTE: We choose the last outcome for multiple shots.
        outcome = outcomes[-1]

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
            @ (outcome - r_measured)
        )

        evolved_mean = np.zeros(shape=(2 * d, ))
        evolved_mean[outer_indices] = evolved_r_A

        evolved_cov = np.identity(2 * d) * constants.HBAR
        evolved_cov[np.ix_(outer_indices, outer_indices)] = evolved_rho_outer

        self.mean = evolved_mean
        self.cov = evolved_cov

        return outcomes

        return outcome

    def _apply_particle_number_measurement(self, *, cutoff, modes):
        if not modes:
            modes = tuple(range(self.d))

        d = len(modes)

        identity = np.identity(d)
        zeros = np.zeros_like(identity)
        X = np.block(
            [
                [zeros, identity],
                [identity, zeros]
            ]
        )

        husimi = self.reduced(modes).husimi

        A = X @ ((np.identity(2 * d) - np.linalg.inv(husimi)))

        acc_modes = []

        for index in range(1, len(modes)):
            acc_modes.append(modes[:index])

        outcome = []

        previous_probability = 1

        for _ in repeat(None, len(modes)):

            probabilities = []

            for n in range(cutoff + 1):
                reduction_indices = outcome + [n]

                A_reduced = block_reduce(A, reduction_indices=reduction_indices)

                probability = (
                    hafnian(A_reduced) / (
                        np.prod(factorial(reduction_indices))
                        * np.sqrt(np.linalg.det(husimi))
                    )
                ).real / previous_probability

                previous_probability = probability

                probabilities.append(probability)

            choice = random.choices(
                population=range(len(probabilities)),
                weights=probabilities,
            )[0]

            outcome.append(choice)

        return outcome
