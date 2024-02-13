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

from typing import Tuple, Mapping

import random
import numpy as np

from .state import FockState

from piquasso.instructions import gates

from piquasso._math.decompositions import euler

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result

from piquasso._backends.fock.calculations import calculate_state_index_matrix_list


def vacuum(state: FockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(
    state: FockState, instruction: gates._PassiveLinearGate, shots: int
) -> Result:
    operator: np.ndarray = instruction._get_passive_block(
        state._calculator, state._config
    )

    _apply_passive_linear(state, operator, instruction.modes)

    return Result(state=state)


def _apply_passive_linear(state, interferometer, modes):
    fock_operator = state._space.get_passive_fock_operator(
        interferometer,
        modes=modes,
        d=state._space.d,
    )

    state._density_matrix = (
        fock_operator @ state._density_matrix @ fock_operator.conjugate().transpose()
    )


def particle_number_measurement(
    state: FockState, instruction: Instruction, shots: int
) -> Result:
    reduced_state = state.reduced(instruction.modes)

    probability_map = reduced_state.fock_probabilities_map

    samples = random.choices(
        population=list(probability_map.keys()),
        weights=list(probability_map.values()),
        k=shots,
    )

    # NOTE: We choose the last sample for multiple shots.
    sample = samples[-1]

    normalization = _get_normalization(probability_map, sample)

    _project_to_subspace(
        state=state,
        subspace_basis=sample,
        modes=instruction.modes,
        normalization=normalization,
    )

    return Result(state=state, samples=samples)


def _get_normalization(
    probability_map: Mapping[Tuple[int, ...], float], sample: Tuple[int, ...]
) -> float:
    return 1 / probability_map[sample]


def _project_to_subspace(
    state: FockState,
    *,
    subspace_basis: Tuple[int, ...],
    modes: Tuple[int, ...],
    normalization: float
) -> None:
    projected_density_matrix = _get_projected_density_matrix(
        state=state,
        subspace_basis=subspace_basis,
        modes=modes,
    )

    state._density_matrix = projected_density_matrix * normalization


def _get_projected_density_matrix(
    state: FockState, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
) -> np.ndarray:
    new_density_matrix = state._get_empty()

    index = state._space.get_projection_operator_indices(
        subspace_basis=subspace_basis,
        modes=modes,
    )

    new_density_matrix[index] = state._density_matrix[index]

    return new_density_matrix


def create(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_creation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return Result(state=state)


def annihilate(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_annihilation_operator(instruction.modes)

    state._density_matrix = operator @ state._density_matrix @ operator.transpose()

    state.normalize()

    return Result(state=state)


def kerr(state: FockState, instruction: Instruction, shots: int) -> Result:
    xi = instruction._all_params["xi"]

    mode = instruction.modes[0]

    for index, (basis, dual_basis) in state._space.operator_basis:
        number = basis[mode]
        dual_number = dual_basis[mode]

        coefficient = np.exp(1j * xi * (number**2 - dual_number**2))

        state._density_matrix[index] *= coefficient

    return Result(state=state)


def cubic_phase(state: FockState, instruction: Instruction, shots: int) -> Result:
    gamma = instruction._all_params["gamma"]
    hbar = state._config.hbar

    matrix = state._space.get_single_mode_cubic_phase_operator(
        gamma=gamma, hbar=hbar, calculator=state._calculator
    )
    _apply_active_gate_matrix_to_state(state, matrix, instruction.modes[0])

    state.normalize()

    return Result(state=state)


def _apply_active_gate_matrix_to_state(
    state: FockState,
    matrix: np.ndarray,
    mode: int,
) -> None:
    density_matrix = state._density_matrix
    space = state._space
    auxiliary_subspace = state._get_subspace(state.d - 1)

    state_index_matrix_list = calculate_state_index_matrix_list(
        space, auxiliary_subspace, mode
    )
    density_matrix = _calculate_density_matrix_after_apply_active_gate(
        density_matrix, matrix, state_index_matrix_list
    )

    state._density_matrix = density_matrix


def _calculate_density_matrix_after_apply_active_gate(
    density_matrix, matrix, state_index_matrix_list
):
    """
    Applies the active gate matrix to the density matrix specified by
    `state_index_matrix_list`.
    """

    new_density_matrix = np.empty_like(density_matrix, dtype=density_matrix.dtype)

    cutoff = len(state_index_matrix_list)

    for ket in range(cutoff):
        state_index_matrix_ket = state_index_matrix_list[ket]
        limit_ket = state_index_matrix_ket.shape[0]
        sliced_matrix_ket = matrix[:limit_ket, :limit_ket]

        for bra in range(max(ket + 1, cutoff)):
            state_index_matrix_bra = state_index_matrix_list[bra]
            limit_bra = state_index_matrix_bra.shape[0]
            sliced_matrix_bra = matrix[:limit_bra, :limit_bra]

            for col_ket in range(state_index_matrix_ket.shape[1]):
                for col_bra in range(state_index_matrix_bra.shape[1]):
                    index = np.ix_(
                        state_index_matrix_ket[:, col_ket],
                        state_index_matrix_bra[:, col_bra].T,
                    )
                    partial_result = np.einsum(
                        "ij,jk,kl->il",
                        sliced_matrix_ket,
                        density_matrix[index],
                        sliced_matrix_bra.T.conj(),
                    )

                    new_density_matrix[index] = partial_result

                    if bra < ket:
                        # NOTE: We use the fact that the density matrix is self-adjoint.
                        index = np.ix_(
                            state_index_matrix_bra[:, col_bra],
                            state_index_matrix_ket[:, col_ket].T,
                        )

                        new_density_matrix[index] = partial_result.conj().T

    return new_density_matrix


def cross_kerr(state: FockState, instruction: Instruction, shots: int) -> Result:
    modes = instruction.modes
    xi = instruction._all_params["xi"]

    for index, (basis, dual_basis) in state._space.operator_basis:
        coefficient = np.exp(
            1j
            * xi
            * (
                basis[modes[0]] * basis[modes[1]]
                - dual_basis[modes[0]] * dual_basis[modes[1]]
            )
        )

        state._density_matrix[index] *= coefficient

    return Result(state=state)


def displacement(state: FockState, instruction: Instruction, shots: int) -> Result:
    r = instruction._all_params["r"]
    phi = instruction._all_params["phi"]
    mode = instruction.modes[0]

    matrix = state._space.get_single_mode_displacement_operator(r=r, phi=phi)

    _apply_active_gate_matrix_to_state(state, matrix, mode=mode)

    state.normalize()

    return Result(state=state)


def squeezing(state: FockState, instruction: Instruction, shots: int) -> Result:
    r = instruction._all_params["r"]
    phi = instruction._all_params["phi"]
    mode = instruction.modes[0]

    matrix = state._space.get_single_mode_squeezing_operator(
        r=r,
        phi=phi,
    )

    _apply_active_gate_matrix_to_state(state, matrix, mode=mode)

    state.normalize()

    return Result(state=state)


def linear(
    state: FockState, instruction: gates._ActiveLinearGate, shots: int
) -> Result:
    calculator = state._calculator
    modes = instruction.modes

    np = calculator.np

    passive_block = instruction._get_passive_block(state._calculator, state._config)
    active_block = instruction._get_active_block(state._calculator, state._config)

    symplectic = calculator.block(
        [
            [passive_block, active_block],
            [np.conj(active_block), np.conj(passive_block)],
        ],
    )

    unitary_last, squeezings, unitary_first = euler(symplectic, calculator)

    _apply_passive_linear(state, unitary_first, modes)

    for mode, r in zip(instruction.modes, squeezings):
        matrix = state._space.get_single_mode_squeezing_operator(r=r, phi=0.0)
        _apply_active_gate_matrix_to_state(state, matrix, mode)

    _apply_passive_linear(state, unitary_last, modes)

    state.normalize()

    return Result(state=state)


def density_matrix_instruction(
    state: FockState, instruction: Instruction, shots: int
) -> Result:
    _add_occupation_number_basis(state, **instruction.params)

    return Result(state=state)


def _add_occupation_number_basis(
    state: FockState,
    *,
    ket: Tuple[int, ...],
    bra: Tuple[int, ...],
    coefficient: complex
) -> None:
    index = state._space.index(ket)
    dual_index = state._space.index(bra)

    state._density_matrix[index, dual_index] = coefficient
