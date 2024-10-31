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

from typing import Optional, TYPE_CHECKING

from piquasso._math.validations import all_in_interval
from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric
from piquasso._math.transformations import (
    from_xxpp_to_xpxp_transformation_matrix,
)
from piquasso._math.combinatorics import sort_and_get_parity

from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.connector import BaseConnector

from ._misc import (
    validate_fermionic_gaussian_hamiltonian,
    get_fermionic_hamiltonian,
    get_omega,
    tensor_product,
)

if TYPE_CHECKING:
    import numpy as np


class GaussianState(State):
    r"""A fermionic Gaussian state."""

    def __init__(
        self,
        d: int,
        connector: BaseConnector,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(connector=connector, config=config)

        self._d = d

        # The upper left part of the correlation matrix, i.e., `\Gamma^{a^\dagger a}`.
        self._D = self._np.empty(shape=(d, d), dtype=self._config.complex_dtype)
        # The upper right part of the correlation matrix, i.e.,
        # `\Gamma^{a^\dagger a^\dagger}`.
        self._E = self._np.empty(shape=(d, d), dtype=self._config.complex_dtype)

    @property
    def d(self):
        return self._d

    @property
    def covariance_matrix(self):
        r"""The covariance matrix.
        A fermionic Gaussian state can be fully characterized by its covariance matrix
        defined by

        .. math::

            \Sigma_{ij} := -i \operatorname{Tr} (\rho [\mathbf{m}_i, \mathbf{m}_j]) / 2.

        The covariance matrix is a real-valued, skew-symmetric matrix.
        """

        d = self.d
        np = self._connector.np

        ident = np.identity(d)

        D = self._D
        E = self._E

        two_D_plus_E = 2 * (D + E)
        two_D_minus_E = 2 * (D - E)

        return np.block(
            [
                [two_D_plus_E.imag, -two_D_minus_E.real + ident],
                [two_D_plus_E.real - ident, two_D_minus_E.imag],
            ]
        )

    @covariance_matrix.setter
    def covariance_matrix(self, input_covariance_matrix):
        d = self.d

        np = self._connector.np

        ident = np.identity(d)

        two_D_plus_E_real = input_covariance_matrix[d:, :d] + ident
        two_D_plus_E_imag = input_covariance_matrix[:d, :d]

        two_D_minus_E_real = -input_covariance_matrix[:d, d:] + ident
        two_D_minus_E_imag = input_covariance_matrix[d:, d:]

        D_plus_E_over_2 = (two_D_plus_E_real + 1j * two_D_plus_E_imag) / 4
        D_minus_E_over_2 = (two_D_minus_E_real + 1j * two_D_minus_E_imag) / 4

        self._D = D_plus_E_over_2 + D_minus_E_over_2
        self._E = D_plus_E_over_2 - D_minus_E_over_2

        self.validate()

    @property
    def correlation_matrix(self):
        r"""The correlation matrix in the Dirac basis.

        The correlation matrix is defined by

        .. math::
            \Gamma := \begin{bmatrix}
                \Gamma^{f^\dagger f} & \Gamma^{f^\dagger f^\dagger} \\
                \Gamma^{f f} & \Gamma^{f f^\dagger}
            \end{bmatrix},

        where :math:`\Gamma_{i,j}^{f^\dagger f} = \langle f_i^\dagger f_j \rangle` and
        :math:`\Gamma_{i,j}^{f f} = \langle f_i f_j \rangle`.

        By CAR, we know that
        :math:`\Gamma_{i,j}^{f^\dagger f} = - \overline{\Gamma_{i,j}^{f f^\dagger }}`
        and
        :math:`\Gamma_{i,j}^{f f} = I - \overline{\Gamma_{i,j}^{f^\dagger f^\dagger }}`

        The correlation matrix is a self-adjoint matrix.
        """

        np = self._connector.np

        ident = np.identity(self.d)

        return np.block([[self._D, self._E], [-self._E.conj(), ident - self._D.conj()]])

    @correlation_matrix.setter
    def correlation_matrix(self, input_correlation_matrix):
        d = self.d
        self._D = input_correlation_matrix[:d, :d]
        self._E = input_correlation_matrix[:d, d:]
        self.validate()

    @property
    def maj_correlation_matrix(self):
        """The correlation matrix in the Majorana operator basis."""
        d = self.d
        np = self._connector.np

        ident = np.identity(d)

        D = self._D
        E = self._E

        two_D_plus_E = 2 * (D + E)
        two_D_minus_E = 2 * (D - E)

        return (
            np.block(
                [
                    [
                        ident + 1j * np.imag(two_D_plus_E),
                        1j * (ident - np.real(two_D_minus_E)),
                    ],
                    [
                        1j * (np.real(two_D_plus_E) - ident),
                        ident + 1j * np.imag(two_D_minus_E),
                    ],
                ]
            )
        ) / 2

    def validate(self):
        if not self._config.validate:
            return

        d = self.d

        np = self._connector.fallback_np

        correlation_matrix = self.correlation_matrix

        expected_shape = (2 * d, 2 * d)
        actual_shape = correlation_matrix.shape

        if not actual_shape == expected_shape:
            raise InvalidState(
                f"The correlation matrix has invalid shape.\n"
                f"expected={expected_shape} "
                f"actual={actual_shape}"
            )

        if not np.allclose(
            correlation_matrix[:d, :d],
            np.identity(d) - correlation_matrix[d:, d:].conj(),
        ) or not np.allclose(
            correlation_matrix[:d, d:], -correlation_matrix[d:, :d].conj()
        ):
            raise InvalidState("The correlation matrix is not physical.")

        if not is_selfadjoint(correlation_matrix[:d, :d]) or not is_skew_symmetric(
            correlation_matrix[:d, d:]
        ):
            raise InvalidState("The correlation matrix is not self-adjoint.")

        eigvals = np.linalg.eigvalsh(correlation_matrix)

        if not all_in_interval(eigvals, 0.0, 1.0):
            raise InvalidState("The correlation matrix has invalid eigenvalues.")

    @property
    def fock_probabilities(self) -> "np.ndarray":
        d = self.d
        probabilities = []

        fallback_np = self._connector.fallback_np

        for i in range(2**d):
            occupation_numbers = fallback_np.empty(d, dtype=int)

            for j in range(d):
                occupation_numbers[d - j - 1] = i % 2
                i = i // 2

            probabilities.append(
                self.get_particle_detection_probability(occupation_numbers)
            )

        return self._connector.np.array(probabilities)

    def get_particle_detection_probability(
        self, occupation_number: "np.ndarray"
    ) -> float:
        basis_state = self.__class__(
            d=self._d, connector=self._connector, config=self._config
        )
        basis_state._set_occupation_numbers(occupation_number)

        return self.overlap(basis_state)

    def mean_particle_numbers(self, modes):
        """Returns the mean particle numbers on the specified modes."""
        np = self._connector.np

        return np.real(np.diag(self._D)[modes,])

    def _set_parent_hamiltonian(self, parent_hamiltonian):
        """Constructs the Gaussian state using its parent Hamiltonian."""

        if self._config.validate:
            validate_fermionic_gaussian_hamiltonian(parent_hamiltonian)

        d = len(parent_hamiltonian) // 2

        np = self._connector.np

        self.correlation_matrix = np.linalg.inv(
            np.identity(2 * d) + self._connector.expm(2 * parent_hamiltonian)
        )

    def get_parent_hamiltonian(self):
        r"""Calculates the parent Hamiltonian.

        When the correlation matrix is not singular, the density matrix is

        .. math::
            \rho = e^{\hat{H}} / \operatorname{Tr} e^{\hat{H}},

        where :math:`\hat{H}` is the parent hamiltonian (see
        :meth:`get_parent_hamiltonian`) given by

        .. math::
            \hat{H} = \mathbf{f}^\dagger H \mathbf{f}, \\\\

            H = \begin{bmatrix}
                A & -\overline{B} \\
                B & -\overline{A}
            \end{bmatrix}.

        where :math:`A^\dagger = A` and :math:`B^T = - B`.

        Raises:
            PiquassoException: When the correlation matrix is singular.
        """
        d = self.d
        np = self._connector.np

        if (
            self._config.validate
            and np.linalg.matrix_rank(self.correlation_matrix) != 2 * self.d
        ):
            raise PiquassoException(
                "Cannot calculate parent Hamiltonian, since the correlation matrix is "
                "singular."
            )

        inverse_correlation_matrix = np.linalg.inv(self.correlation_matrix)

        parent_hamiltonian = (
            self._connector.logm(inverse_correlation_matrix - np.identity(2 * d)) / 2
        )

        return parent_hamiltonian

    @property
    def density_matrix(self) -> "np.ndarray":
        r"""Density matrix in the lexicographic ordering.

        When applicable, the density matrix is just

        .. math::
            \rho = e^{\hat{H}} / \operatorname{Tr} e^{\hat{H}},

        where :math:`\hat{H}` is the parent hamiltonian (see
        :meth:`get_parent_hamiltonian`) given by

        .. math::
            \hat{H} = \mathbf{f}^\dagger H \mathbf{f}, \\\\
            H = \begin{bmatrix}
                A & -\overline{B} \\
                B & -\overline{A}
            \end{bmatrix}.

        where :math:`A^\dagger = A` and :math:`B^T = - B`.

        Note:
            The density matrix is returned in matrix form here. This will probably
            change in the future to tensor form.
        """

        d = self.d
        connector = self._connector
        np = self._connector.np

        blocks, O = self._connector.schur(self.covariance_matrix)

        etas = np.diag(blocks, 1)[::2].copy()

        # NOTE: This would be equivalent to decomposing the covariance matrix in the
        # xpxp basis, but we originally have it in the xxpp basis.
        T = from_xxpp_to_xpxp_transformation_matrix(self.d)
        O = T @ O

        is_special_orthogonal: bool = np.isclose(np.linalg.det(O), 1.0)

        if not is_special_orthogonal:
            # NOTE: If the orthogonal coming from the real Schur decomposition is not
            # special, then there is no corresponding real logarithm from which we can
            # calculate the gate hamiltonian. To fix this, we can just, flip the
            # decomposition.
            flip = connector.block_diag(
                *([np.array([[0, 1], [1, 0]])] + [np.identity(2)] * (d - 1))
            )
            O = O @ flip
            etas = connector.assign(etas, 0, -etas[0])

        # 1. Calculate diagonalized density matrix
        nu = (1 + etas) / 2

        single_mode_dms = []

        for i in range(self.d):
            single_mode_dms.append(np.array([[nu[i], 0], [0, 1 - nu[i]]]))

        rho_D = tensor_product(single_mode_dms)

        # 2. Unitary which diagonalizes the original density matrix
        omega = get_omega(self.d, connector)

        basis_transformation = T @ omega

        i2H = (
            basis_transformation.conj().T
            @ connector.real_logm(O)
            @ basis_transformation
        )

        H = i2H / 2j
        bigH = get_fermionic_hamiltonian(H, connector)
        unitary = connector.expm(1j * bigH)

        return unitary @ rho_D @ unitary.conj().T

    def _set_occupation_numbers(self, occupation_numbers):
        dtype = self._config.dtype
        np = self._connector.np

        plus_minus = np.diag(1 - 2 * np.array(occupation_numbers))

        zeros = np.zeros_like(plus_minus)

        self.covariance_matrix = np.block(
            [[zeros, plus_minus], [-plus_minus, zeros]]
        ).astype(dtype)

    @classmethod
    def _from_representation(
        cls,
        *,
        D: "np.ndarray",
        E: "np.ndarray",
        config: Config,
        connector: BaseConnector,
    ) -> "GaussianState":
        obj = cls(d=len(D), connector=connector, config=config)

        obj._D = D
        obj._E = E

        return obj

    def reduced(self, modes):
        """Reduces the state to a subsystem on the specified modes."""

        index = self._connector.fallback_np.ix_(modes, modes)

        return self._from_representation(
            D=self._D[index],
            E=self._E[index],
            config=self._config,
            connector=self._connector,
        )

    def get_majorana_monomial_expectation_value(self, indices):
        r"""Calculates Majorana monomial expectation values using Wick's theorem.

        The monomial indices are understood in the xxpp-ordering.

        Args:
            indices:
                Indices of the Majorana operators in any order with possible with
                possible multiplicities.
        """

        fallback_np = self._connector.fallback_np

        indices = fallback_np.array(indices)

        sorted_indices, parity = sort_and_get_parity(indices)

        unique, counts = fallback_np.unique(sorted_indices, return_counts=True)
        filtered_indices = unique[counts % 2 == 1]

        matrix_index = fallback_np.ix_(filtered_indices, filtered_indices)

        covariance_matrix_reduced = self.covariance_matrix[matrix_index]

        prefactor = parity * 1j ** (len(filtered_indices) // 2)

        return prefactor * self._connector.pfaffian(covariance_matrix_reduced)

    def get_parity_operator_expectation_value(self):
        r"""Calculates the expectation value of the parity operator.

        The parity operator is defined as

        .. math::
            P = i^d m_1 \dots m_{2d} = i^d x_1 \dots x_d p_1 \dots p_d.
        """
        fallback_np = self._connector.fallback_np

        return 1j**self.d * self.get_majorana_monomial_expectation_value(
            fallback_np.arange(2 * self.d)
        )

    def overlap(self, other):
        r"""Calculates the overlap between two fermionic Gaussian states.

        The overlap is the Hilbert-Schmidt inner product of the density matrices.
        """

        np = self._connector.np

        gamma1 = self.covariance_matrix
        gamma2 = other.covariance_matrix

        ident = np.identity(len(gamma1))

        ret2 = np.sqrt(np.linalg.det((ident - gamma1 @ gamma2) / 2))

        return ret2

    def __eq__(self, other):
        np = self._connector.np
        return (
            type(self) is type(other)
            and np.allclose(self._D, other._D)
            and np.allclose(self._E, other._E)
        )
