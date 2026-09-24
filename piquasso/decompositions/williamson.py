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
            \iff
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
"""


import numpy as np


from piquasso._math.symplectic import xp_symplectic_form
from piquasso._math.transformations import xpxp_to_xxpp_indices
from piquasso.api.connector import BaseConnector


def williamson(matrix: np.ndarray, connector: BaseConnector) -> tuple:
    r"""
    Args:
        matrix (numpy.ndarray): The matrix to decompose.

    Returns:
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

    indices = xpxp_to_xxpp_indices(d)
    basis_change = _rotation_to_positive_above_diagonals(
        block_diagonal_part, connector
    )[:, indices]
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

