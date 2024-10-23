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

import numpy as np

from scipy.optimize import root_scalar

from piquasso._math.symplectic import xp_symplectic_form
from piquasso._math.transformations import from_xxpp_to_xpxp_transformation_matrix

from piquasso.api.exceptions import InvalidParameter
from piquasso.api.connector import BaseConnector


def takagi(matrix, connector, atol=1e-12):
    """Takagi factorization of complex symmetric matrices.

    Note:

        The singular values have to be rounded due to floating point errors.

        The result is not unique in a sense that different result could be obtained
        by different ordering of the singular values.

    References:
    - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.94.062109
    """

    np = connector.np

    V, singular_values, W_adjoint = connector.svd(matrix)

    W = np.conj(W_adjoint).T

    singular_value_multiplicity_indices = []
    singular_value_multiplicity_values = []

    for index, value in enumerate(singular_values):
        matches = np.where(
            np.isclose(value, np.array(singular_value_multiplicity_values), atol=atol)
        )[0]

        if len(matches) == 0:
            singular_value_multiplicity_values.append(value)
            singular_value_multiplicity_indices.append([index])
        else:
            singular_value_multiplicity_indices[matches[0]].append(index)

    diagonal_blocks_for_Q = []

    for indices in singular_value_multiplicity_indices:
        Z = V[:, indices].transpose() @ W[:, indices]

        diagonal_blocks_for_Q.append(connector.sqrtm(Z))

    Q = connector.block_diag(*diagonal_blocks_for_Q)

    return singular_values, V @ np.conj(Q)


def _rotation_to_positive_above_diagonals(block_diagonal_matrix, connector):
    """
    The block diagonal matrix returned by the Schur decomposition in the Williamson
    decomposition needs to be rotated.

    Not doing this we'd still get a valid Williamson decompostion with valid symplectic
    and diagonal matrices, but the symplectic matrix would have complex elements and the
    diagonal matrix would have negative values.
    """

    np = connector.np

    d = len(block_diagonal_matrix) // 2
    identity = np.identity(2)
    rotation = np.rot90(identity)

    return connector.block_diag(
        *[
            (
                identity
                if block_diagonal_matrix[2 * index, 2 * index + 1] > 0
                else rotation
            )
            for index in range(d)
        ]
    )


def williamson(matrix: np.ndarray, connector: BaseConnector) -> tuple:
    r"""
    Decomposes a positive definite matrix with Williamson decomposition, i.e. a
    positive definite :math:`M` is decomposed to

    .. math::

        M = S D S^T,

    where :math:`S \in \operatorname{Sp}(\mathbb{R}, 2d)` is a real symplectic matrix,
    and :math:`D` is a diagonal matrix containing positive values in the diagonal.

    The algorithm works as follows: without loss of generality, one can write the
    symplectic matrix in the form of

    .. math::

        S = M^{1 / 2} K D^{- 1 / 2}

    with :math:`K \in O(2d)`, since then

    .. math::

        M = S D S^T

    by construction. Now we need to find such :math:`K` that the value of :math:`S` is
    symplectic.

    .. math::

        S^T \Omega S = \Omega
            \rightleftarrow
            M^{- 1 / 2} J M^{- 1 / 2} = K D^{- 1 / 2} J D^{- 1 / 2} K^T,

    where

    .. math::

        D^{- 1 / 2} J D^{- 1 / 2}
            =
            \begin{bmatrix}
                0            & \hat{D}^{-1} \\
                \hat{D}^{-1} & 0            \\
            \end{bmatrix}

    is an antisymmetric matrix. We also know that :math:`M^{- 1 / 2} J M^{- 1 / 2}` is
    also antisymmetric. We just need to deduce the orthogonal transformation :math:`K`
    to acquire the symplectic matrix :math:`S`.

    We can use a (real) Schur decomposition to block-diagonalize
    :math:`M^{- 1 / 2} J M^{- 1 / 2}`. Note, that we also rotate the block to have the
    positive values in the above the diagonal to acquire real-valued symplectic matrices
    in the Williamson decomposition. Finally, we can acquire
    :math:`D^{- 1 / 2} J D^{- 1 / 2}` with a simple basis change.

    References:
        - https://math.stackexchange.com/a/1244793

    Args:
        matrix (numpy.ndarray): The matrix to decompose.

    Returns
        tuple: Tuple of the symplectic and diagonal matrices, in this order.
    """
    np = connector.np

    d = len(matrix) // 2

    omega = xp_symplectic_form(d)

    root_matrix = connector.sqrtm(matrix).real
    inverse_root_matrix = np.linalg.inv(root_matrix)

    block_diagonal_part, orthogonal_part = connector.schur(
        inverse_root_matrix @ omega @ inverse_root_matrix,
        output="real",
    )

    basis_change = _rotation_to_positive_above_diagonals(
        block_diagonal_part, connector
    ) @ from_xxpp_to_xpxp_transformation_matrix(d)
    ordered_block_diagonal = basis_change.T @ block_diagonal_part @ basis_change

    inverse_diagonal_matrix = connector.block_diag(
        *(ordered_block_diagonal[:d, d:],) * 2
    )

    root_inverse_diagonal_matrix = np.diag(np.sqrt(np.diag(inverse_diagonal_matrix)))

    symplectic = (
        root_matrix @ orthogonal_part @ basis_change @ root_inverse_diagonal_matrix
    )

    diagonal_matrix = np.diag(1 / np.diag(inverse_diagonal_matrix))

    return symplectic, diagonal_matrix


def decompose_adjacency_matrix_into_circuit(
    adjacency_matrix, mean_photon_number, connector
):
    singular_values, unitary = takagi(adjacency_matrix, connector)

    scaling = _get_scaling(singular_values, mean_photon_number, adjacency_matrix)

    squeezing_parameters = np.arctanh(scaling * singular_values)

    return squeezing_parameters, unitary


def _get_scaling(
    singular_values: np.ndarray, mean_photon_number: float, adjacency_matrix: np.ndarray
) -> float:
    r"""
    For a squeezed state :math:`rho` the mean photon number is calculated by

    .. math::
        \langle n \rangle_\rho = \sum_{i = 0}^d \mathrm{sinh}(r_i)^2

    where :math:`r_i = \mathrm{arctan}(s_i)`, where :math:`s_i` are the singular
    values of the adjacency matrix.
    """

    def mean_photon_number_equation(scaling: float) -> float:
        return (
            sum(
                (scaling * singular_value) ** 2 / (1 - (scaling * singular_value) ** 2)
                for singular_value in singular_values
            )
            / len(singular_values)
            - mean_photon_number
        )

    def mean_photon_number_gradient(scaling: float) -> float:
        return (2.0 / scaling) * np.sum(
            (singular_values * scaling / (1 - (singular_values * scaling) ** 2)) ** 2
        )

    lower_bound = 0.0

    tolerance = 1e-10  # Needed to avoid zero division.

    upper_bound = 1.0 / (max(singular_values) + tolerance)

    result = root_scalar(
        mean_photon_number_equation,
        fprime=mean_photon_number_gradient,
        x0=(lower_bound - upper_bound) / 2.0,
        bracket=(lower_bound, upper_bound),
    )

    if not result.converged:
        raise InvalidParameter(
            f"No scaling found for adjacency matrix: {adjacency_matrix}."
        )

    return result.root


def euler(symplectic, connector):
    np = connector.np
    d = len(symplectic) // 2

    U_orig, R = connector.polar(symplectic, side="left")

    K = np.diag(np.array([1.0] * d + [-1.0] * d))

    H_active = 1j * K @ connector.logm(R)

    Z = 1j * H_active[:d, d:]

    D, U = takagi(Z, connector)

    return U, D, np.conj(U).T @ U_orig[:d, :d]
