#
# Copyright 2021-2022 Budapest Quantum Computing Group
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
from itertools import product

import random
import numpy as np
from scipy.special import factorial

from .state import FockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result
from piquasso._math.functions import hermite


def vacuum(state: FockState, instruction: Instruction, shots: int) -> Result:
    state.reset()

    return Result(state=state)


def passive_linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator: np.ndarray = instruction._all_params["passive_block"]

    fock_operator = state._space.get_passive_fock_operator(
        operator,
        modes=instruction.modes,
        d=state._space.d,
        permanent_function=state._config.permanent_function,
    )

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
    xi_vector = instruction._all_params["xi"]

    for mode_index, mode in enumerate(instruction.modes):
        xi = xi_vector[mode_index]

        for index, (basis, dual_basis) in state._space.operator_basis:
            number = basis[mode]
            dual_number = dual_basis[mode]

            coefficient = np.exp(
                1j
                * xi
                * (number * (2 * number + 1) - dual_number * (2 * dual_number + 1))
            )

            state._density_matrix[index] *= coefficient

    return Result(state=state)


def cubic_phase(state: FockState, instruction: Instruction, shots: int) -> Result:
    gamma = instruction._all_params["gamma"]
    hbar = state._config.hbar

    for index, mode in enumerate(instruction.modes):
        operator = state._space.get_single_mode_cubic_phase_operator(
            gamma=gamma[index], hbar=hbar
        )
        embedded_operator = state._space.embed_matrix(
            operator,
            modes=(mode,),
            auxiliary_modes=state._get_auxiliary_modes(modes=(mode,)),
        )
        state._density_matrix = (
            embedded_operator
            @ state._density_matrix
            @ embedded_operator.conjugate().transpose()
        )

        state.normalize()

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


def displacement(state: FockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.abs(instruction._all_params["displacement_vector"])
    angles = np.angle(instruction._all_params["displacement_vector"])

    for index, mode in enumerate(instruction.modes):
        operator = state._space.get_single_mode_displacement_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        embedded_operator = state._space.embed_matrix(
            operator,
            modes=(mode,),
            auxiliary_modes=state._get_auxiliary_modes(modes=(mode,)),
        )

        state._density_matrix = (
            embedded_operator
            @ state._density_matrix
            @ embedded_operator.conjugate().transpose()
        )

        state.normalize()

    return Result(state=state)


def squeezing(state: FockState, instruction: Instruction, shots: int) -> Result:
    amplitudes = np.arccosh(np.diag(instruction._all_params["passive_block"]))
    angles = np.angle(-np.diag(instruction._all_params["active_block"]))

    for index, mode in enumerate(instruction.modes):
        operator = state._space.get_single_mode_squeezing_operator(
            r=amplitudes[index],
            phi=angles[index],
        )

        embedded_operator = state._space.embed_matrix(
            operator,
            modes=(mode,),
            auxiliary_modes=state._get_auxiliary_modes(modes=(mode,)),
        )

        state._density_matrix = (
            embedded_operator
            @ state._density_matrix
            @ embedded_operator.conjugate().transpose()
        )

        state.normalize()

    return Result(state=state)


def linear(state: FockState, instruction: Instruction, shots: int) -> Result:
    operator = state._space.get_linear_fock_operator(
        modes=instruction.modes,
        passive_block=instruction._all_params["passive_block"],
        active_block=instruction._all_params["active_block"],
        permanent_function=state._config.permanent_function,
    )

    state._density_matrix = (
        operator @ state._density_matrix @ operator.conjugate().transpose()
    )

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


def measure_homodyne(state: FockState, instruction: Instruction, shots: int) -> Result:
    modes = instruction.modes
    phi = instruction._all_params["phi"]
    reduced_state = state.reduced(modes=modes)
    scaled_hbar = 1 / reduced_state._config.hbar

    if phi != 0:
        phi_scaled = np.diag(np.exp(1j * np.atleast_1d(phi)))
        fock_operator = reduced_state._space.get_passive_fock_operator(
            phi_scaled,
            modes=instruction.modes,
            d=reduced_state._space.d,
            permanent_function=reduced_state._config.permanent_function,
        )
        reduced_state._density_matrix = (
            fock_operator
            @ reduced_state._density_matrix
            @ fock_operator.conjugate().transpose()
        )
    resolution = 100000
    cutoff = reduced_state._config.cutoff
    quadrature_range = np.linspace(-cutoff, cutoff, resolution)
    scaled_quadrature_range = np.sqrt(1 / state._config.hbar) * quadrature_range

    # Calculating physicist's Hermite polynomials

    hvalues = []
    for n in range(cutoff):
        hvalues.append(hermite(scaled_quadrature_range, n))

    hermite_matrix = np.zeros((cutoff, cutoff, resolution))
    for n, m in product(range(cutoff), repeat=2):
        hermite_matrix[n][m] = (
            1.0
            / np.sqrt(2**n * factorial(n) * 2**m * factorial(m))
            * hvalues[n]
            * hvalues[m]
        )
    hermite_values = np.expand_dims(reduced_state.density_matrix, -1) * np.expand_dims(
        hermite_matrix, 0
    )

    pdf_rho = (
        (
            np.sum(hermite_values, axis=(1, 2))
            * (scaled_hbar / np.pi) ** 0.5
            * np.exp(-scaled_hbar * quadrature_range**2)
            * (quadrature_range[1] - quadrature_range[0])
        )
        .flatten()
        .real
    )

    pdf_rho /= np.sum(pdf_rho)

    pdf_rho[np.abs(pdf_rho) < 1e-11] = 0

    homodyne_result = []
    for _ in range(shots):
        homodyne_result.append(
            quadrature_range[np.where(np.random.multinomial(1, pdf_rho) == 1)[0]]
        )
    # The remaining state is not collapsed but the reduced density matrix
    return Result(state=reduced_state, samples=homodyne_result)
