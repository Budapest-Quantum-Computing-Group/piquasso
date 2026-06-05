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


from fractions import Fraction
from functools import lru_cache
from itertools import repeat

from piquasso.api.branch import Branch
from piquasso.api.exceptions import InvalidParameter
from piquasso.api.instruction import Instruction

from piquasso.instructions.gates import _PassiveLinearGate, Squeezing2
from piquasso.fermionic.instructions import IsingXX

from piquasso._math.validations import all_zero_or_one
from piquasso._math.transformations import (
    xxpp_to_xpxp_indices,
)

from piquasso._math.indices import double_modes

from .._utils import validate_fermionic_gaussian_hamiltonian
from .state import GaussianState

import typing

if typing.TYPE_CHECKING:
    import numpy as np
    from typing import Tuple, List


def vacuum(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Branch]":
    """Sets the vacuum state."""

    np = state._connector.np

    state._set_occupation_numbers(np.zeros(state.d, dtype=state._config.dtype))

    return [Branch(state=state)]


def state_vector(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Branch]":
    """Sets an arbitrary occupation number state."""

    fallback_np = state._connector.fallback_np

    coefficient = instruction._get_all_params(state._connector)["coefficient"]
    if state._can_validate_variable(coefficient) and not fallback_np.isclose(
        coefficient, 1.0
    ):
        raise InvalidParameter(
            "Only 1.0 is permitted as coefficient for the state vector."
        )

    occupation_numbers = fallback_np.array(
        instruction._get_all_params(state._connector)["occupation_numbers"]
    )

    if state._config.validate and not all_zero_or_one(occupation_numbers):
        raise InvalidParameter(
            f"Invalid initial state specified: instruction={instruction}"
        )

    state._set_occupation_numbers(occupation_numbers)

    return [Branch(state=state)]


def parent_hamiltonian(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Branch]":
    state._set_parent_hamiltonian(instruction.params["hamiltonian"])

    return [Branch(state=state)]


def gaussian_hamiltonian(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Branch]":
    hamiltonian = instruction.params["hamiltonian"]
    modes = instruction.modes

    validate_fermionic_gaussian_hamiltonian(hamiltonian)
    h = _transform_to_majorana_basis(hamiltonian, state._connector)

    evolved_state = _do_apply_gaussian_hamiltonian(state, h, modes)

    return [Branch(state=evolved_state)]


def _transform_to_majorana_basis(H, connector):
    r"""Transform quadratic gate Hamiltonian to Majorana basis.

    Consider a gate unitary of the form

    .. math::
        U = e^{i \hat{H}},

    where

    .. math::
        \hat{H} &= \mathbf{f} H \mathbf{f}^\dagger, \\\\
        H &= \begin{bmatrix}
            A & -\overline{B} \\
            B & -\overline{A}
        \end{bmatrix}.

    Changing to Majorana basis, one can rewrite this as

    .. math::
        \hat{H} &= i \sum_{j,k=1}^{2d} h_{jk} m_j m_k,

    where :math:`m_j` denotes the Majorana operators

    .. math::
        m_{2k} &:= f_k + f_k^\dagger, \\\\
        p_{2k+1} &:= -i (f_k - f_k^\dagger).

    Returns:
        np.ndarray: The matrix `h`.
    """

    np = connector.np
    fallback_np = connector.fallback_np

    small_d = len(H) // 2

    A = H[small_d:, small_d:]
    B = H[:small_d, small_d:]

    A_plus_B = A + B
    A_minus_B = A - B
    indices = xxpp_to_xpxp_indices(small_d)

    return (
        np.block(
            [
                [A_plus_B.imag, A_plus_B.real],
                [-A_minus_B.real, A_minus_B.imag],
            ]
        )[fallback_np.ix_(indices, indices)]
        / 2
    )


def _do_apply_gaussian_hamiltonian(
    state: GaussianState, h: "np.ndarray", modes: "Tuple[int, ...]"
) -> GaussianState:
    connector = state._connector
    np = connector.np
    fallback_np = connector.fallback_np

    d = state._d

    doubled_modes = double_modes(fallback_np.array(modes))

    SO = connector.embed_in_identity(
        connector.expm(-4 * h), np.ix_(doubled_modes, doubled_modes), 2 * d
    )

    state.covariance_matrix = SO @ state.covariance_matrix @ SO.T

    return state


def passive_linear_gate(
    state: GaussianState, instruction: _PassiveLinearGate, shots: int
) -> "List[Branch]":
    r"""Applies a passive linear gate to a fermionic Gaussian state.

    The transformation is equivalent to

    .. math::
        \Gamma^{a^\dagger a} &\mapsto \overline{U} \Gamma^{a^\dagger a} U^T, \\\\
        \Gamma^{a^\dagger a^\dagger} &\mapsto \overline{U} \Gamma^{a^\dagger a^\dagger} U^\dagger,

    where :math:`U` is the unitary corresponding to the passive linear gate in the Dirac
    basis.
    """  # noqa: E501

    unitary = instruction._get_passive_block(state._connector, state._config)
    modes = instruction.modes

    d = state._d

    connector = state._connector

    fallback_np = connector.fallback_np

    all_modes = fallback_np.arange(d)

    select_columns = fallback_np.ix_(all_modes, modes)
    select_rows = fallback_np.ix_(modes, all_modes)

    state._D = connector.assign(
        state._D, select_columns, state._D[select_columns] @ unitary.T
    )
    state._D = connector.assign(
        state._D, select_rows, unitary.conj() @ state._D[modes, :]
    )

    state._E = connector.assign(
        state._E, select_columns, state._E[select_columns] @ unitary.T.conj()
    )
    state._E = connector.assign(
        state._E, select_rows, unitary.conj() @ state._E[select_rows]
    )

    return [Branch(state=state)]


def squeezing2(
    state: GaussianState, instruction: Squeezing2, shots: int
) -> "List[Branch]":
    connector = state._connector
    np = connector.np
    complex_dtype = state._D.dtype

    modes = instruction.modes
    r = instruction.params["r"]
    phi = instruction.params["phi"]

    r_cos_phi = r * np.cos(phi)
    r_sin_phi = r * np.sin(phi)

    h_active = (
        np.array(
            [
                [r_cos_phi, r_sin_phi],
                [r_sin_phi, -r_cos_phi],
            ],
            dtype=complex_dtype,
        )
        / 8
    )

    zeros = np.zeros_like(h_active)

    h = np.block([[zeros, h_active], [-h_active, zeros]])

    evolved_state = _do_apply_gaussian_hamiltonian(state, h, modes)

    return [Branch(state=evolved_state)]


def ising_XX(state: GaussianState, instruction: IsingXX, shots: int) -> "List[Branch]":
    phi = instruction.params["phi"]

    connector = state._connector

    np = connector.np

    modes = instruction.modes

    h = np.zeros((4, 4), dtype=state._config.dtype)

    h = connector.assign(h, (1, 2), -phi / 2)
    h = connector.assign(h, (2, 1), phi / 2)

    evolved_state = _do_apply_gaussian_hamiltonian(state, h, modes)

    return [Branch(state=evolved_state)]


def particle_number_measurement(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Branch]":
    """Performs a particle number measurement on a fermionic Gaussian state.

    Since fermions obey the Pauli exclusion principle, each measured mode yields an
    occupation number of either 0 or 1.
    """
    samples = _generate_particle_number_samples(state, instruction, shots)

    return [
        Branch(state=None, outcome=outcome, frequency=Fraction(1, shots))
        for outcome in samples
    ]


def _generate_particle_number_samples(
    state: GaussianState, instruction: Instruction, shots: int
) -> "List[Tuple[int, ...]]":
    r"""Generates particle number samples using mode-by-mode (chain-rule) sampling.

    To avoid computing all :math:`2^n` output probabilities, the joint distribution is
    factorized as

    .. math::
        p(n_1, \dots, n_d) = \prod_{k=1}^d p(n_k \mid n_1, \dots, n_{k-1}),

    where each conditional is obtained from reduced-state particle detection
    probabilities, see
    :meth:`~piquasso.fermionic.gaussian.state.GaussianState.get_particle_detection_probability`.
    """  # noqa: E501
    rng = state._config.rng
    modes = instruction.modes
    fallback_np = state._connector.fallback_np

    @lru_cache(state._config.cache_size)
    def get_probability(
        subspace_modes: "Tuple[int, ...]", occupation_numbers: "Tuple[int, ...]"
    ) -> float:
        reduced_state = state.reduced(subspace_modes)

        with fallback_np.errstate(invalid="ignore"):
            probability = float(
                fallback_np.real(
                    reduced_state.get_particle_detection_probability(
                        fallback_np.array(occupation_numbers)
                    )
                )
            )

        # `get_particle_detection_probability` may return `nan` (or a tiny negative
        # value) when the true probability is essentially zero, since the determinant
        # under the square root can dip slightly below zero numerically.
        if fallback_np.isnan(probability) or probability < 0.0:
            return 0.0

        return probability

    samples = []

    for _ in repeat(None, shots):
        sample: "List[int]" = []

        previous_probability = 1.0

        for mode_index in range(1, len(modes) + 1):
            subspace_modes = tuple(modes[:mode_index])

            occupation_numbers = tuple(sample + [0])

            probability = get_probability(
                subspace_modes=subspace_modes, occupation_numbers=occupation_numbers
            )
            conditional_probability = probability / previous_probability

            conditional_probability = min(max(conditional_probability, 0.0), 1.0)

            if rng.uniform() < conditional_probability:
                sample.append(0)
                previous_probability *= conditional_probability
            else:
                sample.append(1)
                previous_probability *= 1 - conditional_probability

        samples.append(tuple(sample))

    return samples
