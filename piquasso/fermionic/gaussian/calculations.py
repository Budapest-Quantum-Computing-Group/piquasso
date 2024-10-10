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


from piquasso.api.result import Result
from piquasso.api.exceptions import InvalidParameter
from piquasso.api.instruction import Instruction

from piquasso.instructions.gates import _PassiveLinearGate

from piquasso._math.validations import all_zero_or_one

from ._misc import validate_fermionic_gaussian_hamiltonian
from .state import GaussianState


def vacuum(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    """Sets the vacuum state."""

    np = state._connector.np

    state._set_occupation_numbers(np.zeros(state.d, dtype=state._config.dtype))

    return Result(state=state)


def state_vector(state: GaussianState, instruction: Instruction, shots: int) -> Result:
    """Sets an arbitrary occupation number state."""

    fallback_np = state._connector.fallback_np

    coefficient = instruction._all_params["coefficient"]
    if state._config.validate and not fallback_np.isclose(coefficient, 1.0):
        raise InvalidParameter(
            "Only 1.0 is permitted as coefficient for the state vector."
        )

    occupation_numbers = fallback_np.array(
        instruction._all_params["occupation_numbers"]
    )

    if state._config.validate and not all_zero_or_one(occupation_numbers):
        raise InvalidParameter(
            f"Invalid initial state specified: instruction={instruction}"
        )

    state._set_occupation_numbers(occupation_numbers)

    return Result(state=state)


def parent_hamiltonian(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    state._set_parent_hamiltonian(instruction.params["hamiltonian"])

    return Result(state=state)


def gaussian_hamiltonian(
    state: GaussianState, instruction: Instruction, shots: int
) -> Result:
    H = instruction.params["hamiltonian"]

    validate_fermionic_gaussian_hamiltonian(H)

    np = state._connector.np

    d = state.d

    A = H[d:, d:]
    B = H[:d, d:]

    A_plus_B = A + B
    A_minus_B = A - B

    h = np.block(
        [
            [A_plus_B.imag, A_plus_B.real],
            [-A_minus_B.real, A_minus_B.imag],
        ]
    )

    SO = state._connector.expm(-2 * h)

    state.covariance_matrix = SO @ state.covariance_matrix @ SO.T

    return Result(state=state)


def passive_linear_gate(
    state: GaussianState, instruction: _PassiveLinearGate, shots: int
) -> Result:
    r"""Applies a passive linear gate to a fermionic Gaussian state.

    The transformation is equivalent to

    .. math::
        \Gamma^{a^\dagger a} &\mapsto U \Gamma^{a^\dagger a} U^\dagger, \\\\
        \Gamma^{a^\dagger a^\dagger} &\mapsto U \Gamma^{a^\dagger a^\dagger} U^T,

    where :math:`U` is the unitary corresponding to the passive linear gate in the Dirac
    basis.
    """

    unitary = instruction._get_passive_block(state._connector, state._config)
    modes = instruction.modes

    d = state._d

    connector = state._connector

    fallback_np = connector.fallback_np

    all_modes = fallback_np.arange(d)

    select_columns = fallback_np.ix_(all_modes, modes)
    select_rows = fallback_np.ix_(modes, all_modes)

    state._D = connector.assign(
        state._D, select_columns, state._D[select_columns] @ unitary.conj().T
    )
    state._D = connector.assign(state._D, select_rows, unitary @ state._D[modes, :])

    state._E = connector.assign(
        state._E, select_columns, state._E[select_columns] @ unitary.T
    )
    state._E = connector.assign(state._E, select_rows, unitary @ state._E[select_rows])

    return Result(state=state)
