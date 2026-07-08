#
# Copyright 2021-2026 Budapest Quantum Computing Group
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


from piquasso.api.exceptions import InvalidParameter


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

        D, Q = connector.schur(Z)
        diags = np.diag(D)

        # NOTE: It is not mentioned in the cited paper, but it does matter which square
        # root you take here. If the square root is not the "canonical" one, the
        # decomposition might not yield the original matrix.
        angles_mod = np.mod(np.angle(diags), 2 * np.pi)  # phases in [0, 2\pi)
        sqrt_diags = np.sqrt(np.abs(diags)) * np.exp(1j * angles_mod / 2)

        sqrt_Z = Q @ np.diag(sqrt_diags) @ np.conj(Q).T

        diagonal_blocks_for_Q.append(sqrt_Z)

    Q = connector.block_diag(*diagonal_blocks_for_Q)

    return singular_values, V @ np.conj(Q)




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
