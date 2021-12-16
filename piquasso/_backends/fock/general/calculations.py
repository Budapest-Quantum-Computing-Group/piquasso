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

from typing import Tuple, Mapping, List, Any

import random
import numpy as np

from .state import FockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result

from piquasso._math.indices import get_operator_index


def vacuum(state: FockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator: np.ndarray = instruction._all_params["passive_block"]

    index = get_operator_index(instruction.modes)

    embedded_operator = np.identity(state._space.d, dtype=complex)

    embedded_operator[index] = operator

    fock_operator = state._space.get_passive_fock_operator(embedded_operator)

    state._density_matrix = (
        fock_operator @ state._density_matrix @ fock_operator.conjugate().transpose()
    )

    return Result(state=state)


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
    mode = instruction.modes[0]
    xi = instruction._all_params["xi"]

    for index, (basis, dual_basis) in state._space.operator_basis:
        number = basis[mode]
        dual_number = dual_basis[mode]

        coefficient = np.exp(
            1j * xi * (number * (2 * number + 1) - dual_number * (2 * dual_number + 1))
        )

        state._density_matrix[index] *= coefficient

    return Result(state=state)


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


def linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        cache_size=state._config.cache_size,
        auxiliary_modes=state._get_auxiliary_modes(instruction.modes),
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        displacement=instruction._all_params["displacement_vector"],
    )

    state._density_matrix = (
        operator @ state._density_matrix @ operator.conjugate().transpose()
    )

    state.normalize()

    return Result(state=state)


def generate_squeezing_operator(r: List[Any], phi: List[Any], cutoff_dims: int) -> List:
    """
    This function generates the Squeezing operator following a recursion rule.

    Args:
        r (float): This is the Squeezing amplitude. Typically this value can be
            negative or positive depending on the desired squeezing direction.
            Note:
                Setting :math:`|r|` to higher values will require you to have a higer
                cuttof dimensions.
        phi (float): This is the Squeezing angle. Its ranges are
            :math:`\phi \in [ 0, 2 \pi )`
        cutoff_dims (int): The cutoff of the Fock operator. This value should be
            equal to the cutoff of the Fock state.

    Returns:
        np.ndarray: The constructed Squeezing matrix representing the Fock operator.
    """
    squeezers = []
    for amp, angle in zip(r, phi):
        print(amp, angle)
        sechr = 1.0 / np.cosh(amp)
        A = np.exp(1j * angle) * np.tanh(amp)
        squeezing_matrix = np.empty((cutoff_dims, cutoff_dims), dtype=complex)
        squeezing_matrix[0, 0] = np.sqrt(sechr)
        fock_numbers = np.sqrt(np.arange(cutoff_dims, dtype=complex))

        for m in range(2, cutoff_dims, 2):  # even indices
            squeezing_matrix[m, 0] = (
                -fock_numbers[m - 1]
                / fock_numbers[m]
                * (squeezing_matrix[m - 2, 0] * A)
            )

        for m in range(0, cutoff_dims):
            for n in range(1, cutoff_dims):
                if (m + n) % 2 == 0:  # even indices
                    squeezing_matrix[m, n] = (
                        1
                        / fock_numbers[n]
                        * (
                            (fock_numbers[m] * squeezing_matrix[m - 1, n - 1] * sechr)
                            + (
                                fock_numbers[n - 1]
                                * A.conj()
                                * squeezing_matrix[m, n - 2]
                            )
                        )
                    )
        squeezers.append(squeezing_matrix)

    return squeezers


def single_mode_squeezing(
    state: FockState, instruction: Instruction, shots: int
) -> Result:

    amplitudes = instruction._all_params["r"]
    angles = instruction._all_params["phi"]
    cutoff = state._config.cutoff
    # modes = instruction.modes
    if not isinstance(amplitudes, list):
        amplitudes = [amplitudes]
    if not isinstance(angles, list):
        angles = [angles]

    if state.d == 1:  # TODO: this should be generalized to many qumodes
        S = generate_squeezing_operator(amplitudes, angles, cutoff)[0]
        for operator in S:
            state._density_matrix = operator @ state._density_matrix @ operator.conj().T
            state.normalize()
        return Result(state=state)

    else:
        S = generate_squeezing_operator(amplitudes, angles, cutoff)
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
