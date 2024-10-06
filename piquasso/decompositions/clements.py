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

"""An implementation for the Clements decomposition."""

from typing import List, Tuple, TYPE_CHECKING

from dataclasses import dataclass

from piquasso.api.connector import BaseConnector

from piquasso.instructions.gates import Phaseshifter, Beamsplitter

from piquasso._math.indices import get_operator_index

from piquasso._simulators.connectors import NumpyConnector

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BS:
    """
    Beamsplitter gate, implemented as described in
    `arXiv:1603.08788 <https://arxiv.org/abs/1603.08788>`_.

    Note:
        This means a different beamsplitter gate as defined by
        :class:`~piquasso.instructions.gates.Beamsplitter`.
    """

    modes: Tuple[int, int]
    params: Tuple["np.float64", "np.float64"]


@dataclass
class PS:
    """
    Phaseshifter gate, corresponding to
    :class:`~piquasso.instructions.gates.Phaseshifter`.
    """

    mode: int
    phi: "np.float64"


@dataclass
class Decomposition:
    """
    The data stucture which holds the decomposed angles from the Clements decomposition.

    Example usage::

        decomposition = clements(U, connector=pq.NumpyConnector())

        with pq.Program() as program:
            ...

            for operation in decomposition.first_beamsplitters:
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
                pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

            for operation in decomposition.middle_phaseshifters:
                pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

            for operation in decomposition.last_beamsplitters:
                pq.Q(*operation.modes) | pq.Beamsplitter(-operation.params[0], 0.0)
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=-operation.params[1])

    """

    first_beamsplitters: List[BS]
    middle_phaseshifters: List[PS]
    last_beamsplitters: List[BS]


def instructions_from_decomposition(decomposition):
    """Helper function for using the Clements decomposition.

    This function creates the list of :class:`~piquasso.instructions.gates.Beamsplitter`
    and :class:`~piquasso.instructions.gates.Phaseshifter` instructions to be applied to
    the program, which is equivalent to the decomposed interferometer.

    Example usage::

        decomposition = clements(U, connector=pq.NumpyConnector())

        with pq.Program() as program:
            ...

            program.instructions.extend(instructions_from_decomposition(decomposition))

    Or, one can use it as::

        decomposition = clements(U, connector=pq.NumpyConnector())

        program_with_decomposition = pq.Program(
            instructions=[...] + instructions_from_decomposition(decomposition)
        )

    Args:
        decomposition (Decomposition): Decomposition created by :func:`clements`.

    Returns:
        list: List of beamsplitter and phaseshifter gates to be applied.
    """
    instructions = []

    for bs in decomposition.first_beamsplitters:
        instructions.append(Phaseshifter(bs.params[1]).on_modes(bs.modes[0]))
        instructions.append(Beamsplitter(bs.params[0], 0.0).on_modes(*bs.modes))

    for ps in decomposition.middle_phaseshifters:
        instructions.append(Phaseshifter(ps.phi).on_modes(ps.mode))

    for bs in decomposition.last_beamsplitters:
        instructions.append(Beamsplitter(-bs.params[0], 0.0).on_modes(*bs.modes))
        instructions.append(Phaseshifter(-bs.params[1]).on_modes(bs.modes[0]))

    return instructions


def inverse_clements(
    decomposition: Decomposition, connector: BaseConnector, dtype: "np.dtype"
) -> "np.ndarray":
    """Inverse of the Clements decomposition.

    Returns:
        The unitary matrix corresponding to the interferometer.
    """

    d = len(decomposition.middle_phaseshifters)

    np = connector.np
    interferometer = np.identity(d, dtype=dtype)

    for beamsplitter in decomposition.first_beamsplitters:
        beamsplitter_matrix = _get_embedded_beamsplitter_matrix(
            beamsplitter, d, connector, dtype=dtype
        )

        interferometer = beamsplitter_matrix @ interferometer

    phis = np.empty(d, dtype=interferometer.dtype)

    for phaseshifter in decomposition.middle_phaseshifters:
        phis = connector.assign(phis, phaseshifter.mode, phaseshifter.phi)

    interferometer = np.diag(np.exp(1j * phis)) @ interferometer

    for beamsplitter in decomposition.last_beamsplitters:
        beamsplitter_matrix = np.conj(
            _get_embedded_beamsplitter_matrix(beamsplitter, d, connector, dtype=dtype)
        ).T

        interferometer = beamsplitter_matrix @ interferometer

    return interferometer


def clements(
    U: "np.ndarray",
    connector: BaseConnector,
) -> Decomposition:
    """
    Decomposes the specified unitary matrix by application of beamsplitters
    prescribed by the decomposition.

    Args:
        U (numpy.ndarray): The (square) unitary matrix to be decomposed.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """

    first_beamsplitters = []
    last_beamsplitters = []

    np = connector.np

    d = U.shape[0]

    for column in reversed(range(0, d - 1)):
        if column % 2 == 0:
            operations, U = _apply_direct_beamsplitters(column, U, connector)
            last_beamsplitters.extend(operations)
        else:
            operations, U = _apply_inverse_beamsplitters(column, U, connector)
            first_beamsplitters.extend(operations)

    middle_phasshifters = [
        PS(mode=mode, phi=np.angle(diagonal))
        for mode, diagonal in enumerate(np.diag(U))
    ]

    last_beamsplitters = list(reversed(last_beamsplitters))

    return Decomposition(
        first_beamsplitters=first_beamsplitters,
        last_beamsplitters=last_beamsplitters,
        middle_phaseshifters=middle_phasshifters,
    )


def _apply_direct_beamsplitters(
    column: int, U: "np.ndarray", connector: BaseConnector
) -> tuple:
    """
    Calculates the direct beamsplitters for a given column `column`, and
    applies it to `U`.

    Args:
        column (int): The current column.
    """

    operations = []

    d = U.shape[0]

    dtype = U.dtype

    for j in range(d - 1 - column):
        modes = (column + j, column + j + 1)

        matrix_element_to_eliminate = U[modes[0], j]
        matrix_element_above = -U[modes[1], j]

        angles = _get_angles(
            matrix_element_to_eliminate, matrix_element_above, connector
        )

        operation = BS(modes=modes, params=angles)

        matrix = _get_embedded_beamsplitter_matrix(operation, d, connector, dtype)

        U = matrix @ U

        operations.append(operation)

    return operations, U


def _apply_inverse_beamsplitters(
    column: int, U: "np.ndarray", connector: BaseConnector
) -> tuple:
    """
    Calculates the inverse beamsplitters for a given column `column`, and
    applies it to `U`.

    Args:
        column (int): The current column.
    """

    operations = []

    d = U.shape[0]

    dtype = U.dtype

    for j in reversed(range(d - 1 - column)):
        modes = (j, j + 1)

        i = column + j + 1

        matrix_element_to_eliminate = U[i, modes[1]]
        matrix_element_to_left = U[i, modes[0]]

        angles = _get_angles(
            matrix_element_to_eliminate, matrix_element_to_left, connector
        )

        operation = BS(modes=modes, params=angles)

        beamsplitter = connector.np.conj(
            _get_embedded_beamsplitter_matrix(operation, d, connector, dtype)
        ).T

        U = U @ beamsplitter

        operations.append(operation)

    return operations, U


def _get_angles(matrix_element_to_eliminate, other_matrix_element, connector):
    np = connector.np

    if np.isclose(matrix_element_to_eliminate, 0.0):
        return np.pi / 2, 0.0

    r = other_matrix_element / matrix_element_to_eliminate
    theta = np.arctan(np.abs(r))
    phi = np.angle(r)

    return theta, phi


def _get_embedded_beamsplitter_matrix(
    operation: BS, d: int, connector: BaseConnector, dtype: "np.dtype"
) -> "np.ndarray":
    np = connector.np

    theta, phi = operation.params
    i, j = operation.modes

    c = np.cos(theta).astype(dtype)
    s = np.sin(theta).astype(dtype)

    matrix = np.array(
        [
            [np.exp(1j * phi) * c, -s],
            [np.exp(1j * phi) * s, c],
        ],
        dtype=dtype,
    )

    return connector.embed_in_identity(matrix, get_operator_index((i, j)), d)


def get_weights_from_decomposition(
    decomposition: Decomposition, d: int, connector: BaseConnector
) -> "np.ndarray":
    """Concatenates the weight vector from the angles in the Clements decomposition.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """
    np = connector.np

    dtype = decomposition.middle_phaseshifters[0].phi.dtype
    weights = np.empty(d**2, dtype=dtype)

    index = 0
    for beamsplitter in decomposition.first_beamsplitters:
        weights = connector.assign(weights, index, beamsplitter.params[0])
        index += 1
        weights = connector.assign(weights, index, beamsplitter.params[1])
        index += 1

    for phaseshifter in decomposition.middle_phaseshifters:
        weights = connector.assign(weights, index, phaseshifter.phi)
        index += 1

    for beamsplitter in decomposition.last_beamsplitters:
        weights = connector.assign(weights, index, beamsplitter.params[0])
        index += 1
        weights = connector.assign(weights, index, beamsplitter.params[1])
        index += 1

    return weights


def get_decomposition_from_weights(
    weights: "np.ndarray", d: int, connector: BaseConnector
) -> Decomposition:
    """Puts the data in the weight vector into a Clements decompositon.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """

    fallback_np = connector.fallback_np

    # NOTE: This is tricky: the ordering in the Clements decomposition is not unique,
    # since beamsplitters acting on different modes may commute, and the ordering comes
    # out very ugly after all the Givens rotations. Therefore, it is easier to just
    # create a trivial decomposition, and fill it with the required values (for now).
    decomposition = clements(fallback_np.identity(d), connector=NumpyConnector())

    index = 0

    for beamsplitter in decomposition.first_beamsplitters:
        beamsplitter.params = (weights[index], weights[index + 1])
        index += 2

    for phaseshifter in decomposition.middle_phaseshifters:
        phaseshifter.phi = weights[index]
        index += 1

    for beamsplitter in decomposition.last_beamsplitters:
        beamsplitter.params = (weights[index], weights[index + 1])
        index += 2

    return decomposition


def get_weights_from_interferometer(
    U: "np.ndarray", connector: BaseConnector
) -> "np.ndarray":
    """Creates a vector of weights from the Clements angles."""
    decomposition = clements(U, connector)

    return get_weights_from_decomposition(decomposition, U.shape[0], connector)


def get_interferometer_from_weights(
    weights: "np.ndarray",
    d: int,
    connector: BaseConnector,
    dtype: "np.dtype",
) -> "np.ndarray":
    """Returns the interferometer matrix corresponding to the specified weights.

    It is the inverse of :func:`get_weights_from_interferometer`.
    """
    decomposition = get_decomposition_from_weights(weights, d, connector)

    return inverse_clements(decomposition, connector, dtype)
