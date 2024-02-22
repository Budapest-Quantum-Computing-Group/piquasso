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

from typing import Tuple, List

import numpy as np

from scipy.special import comb

from functools import lru_cache

from .state import BaseFockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result
from piquasso.api.exceptions import InvalidInstruction, InvalidParameter

from piquasso._math.fock import cutoff_cardinality, operator_basis, get_fock_space_basis
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_auxiliary_modes,
    get_index_in_fock_space_array,
    get_index_in_fock_subspace_array,
)


def attenuator(state: BaseFockState, instruction: Instruction, shots: int) -> Result:
    r"""
    Performs the deterministic loss or attenuation channel :math:`C` according to the
    equation

    .. math::
        C(| n \rangle \langle m |) =
            \sum_{k = 0}^{\min(n,m)} \tan(\theta)^{2k} \cos(\theta)^{n + m}
            \sqrt{ {n \choose k} {m \choose k} } | n - k \rangle \langle m - k |.
    """

    modes = instruction.modes

    if len(modes) != 1:
        raise InvalidInstruction(
            f"The instruction should be specified for '{len(modes)}' "
            f"modes: instruction={instruction}"
        )

    mean_thermal_excitation = instruction._all_params["mean_thermal_excitation"]

    if not np.isclose(mean_thermal_excitation, 0.0):
        raise InvalidParameter(
            "Non-zero mean thermal excitation is not supported in this backend. "
            f"mean_thermal_excitation={mean_thermal_excitation}"
        )

    theta = instruction._all_params["theta"]

    space = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    new_state = state._as_mixed()

    new_density_matrix = new_state._get_empty()

    for index, basis in operator_basis(space):
        coefficient = new_state._density_matrix[index]

        ket = np.copy(basis[0])
        bra = np.copy(basis[1])

        n = ket[modes]
        m = bra[modes]

        common_term = coefficient * np.cos(theta) ** (n + m)

        for k in range(min(n, m) + 1):
            ket[(modes,)] -= k
            bra[(modes,)] -= k

            current_ket_index = get_index_in_fock_space(ket)
            current_bra_index = get_index_in_fock_space(bra)

            ket[(modes,)] += k
            bra[(modes,)] += k

            current_index = (current_ket_index, current_bra_index)

            new_density_matrix[current_index] += common_term * (
                np.tan(theta) ** (2 * k) * np.sqrt(comb(n, k) * comb(m, k))
            )

    new_state._density_matrix = new_density_matrix

    return Result(state=new_state)


@lru_cache(maxsize=None)
def calculate_state_index_matrix_list(d, cutoff, mode):
    auxiliary_subspace = get_fock_space_basis(d=d - 1, cutoff=cutoff)

    state_index_matrix_list = []

    indices = [cutoff_cardinality(cutoff=n - 1, d=d - 1) for n in range(1, cutoff + 2)]

    for n in range(cutoff):
        limit = cutoff - n
        limit_range = np.arange(limit)
        subspace_size = indices[n + 1] - indices[n]
        subspace_to_insert = auxiliary_subspace[indices[n] : indices[n + 1]]

        update_on_modes = np.repeat(limit_range[:, None, None], subspace_size, axis=1)

        update_on_aux_modes = np.repeat(subspace_to_insert[None, :, :], limit, axis=0)

        unordered_basis_tensor = np.concatenate(
            [update_on_modes, update_on_aux_modes],
            axis=2,
        )

        reorder_index = np.concatenate(
            [np.arange(1, mode + 1), np.array([0]), np.arange(mode + 1, d)]
        )

        basis_tensor = unordered_basis_tensor[:, :, reorder_index]

        state_index_matrix_list.append(get_index_in_fock_space_array(basis_tensor))

    return state_index_matrix_list


@lru_cache(maxsize=None)
def calculate_interferometer_helper_indices(d, cutoff):
    space = get_fock_space_basis(d=d, cutoff=cutoff)

    indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

    subspace_index_tensor = []

    first_subspace_index_tensor = []
    first_nonzero_index_tensor = []

    sqrt_occupation_numbers_tensor = []
    sqrt_first_occupation_numbers_tensor = []

    identity = np.identity(d, dtype=int)

    repeated = get_index_in_fock_subspace_array(
        np.repeat(space[:, None, :], d, axis=1)
        - np.repeat(identity[None, :, :], len(space), axis=0)
    )

    sqrt_space = np.sqrt(space)
    first_nonzero_space_index = (space != 0).argmax(axis=1)

    space[np.arange(len(space)), first_nonzero_space_index] -= 1
    first_subpace_indices_space = get_index_in_fock_subspace_array(space)
    space[np.arange(len(space)), first_nonzero_space_index] += 1

    sqrt_first_occupation_numbers = np.sqrt(
        space[np.arange(len(space)), first_nonzero_space_index]
    )

    for n in range(2, cutoff):
        subspace_range = np.arange(indices[n - 1], indices[n])
        subspace_index_tensor.append(
            np.mod(repeated[subspace_range], indices[n - 1] - indices[n - 2])
        )

        first_nonzero_index_tensor.append(first_nonzero_space_index[subspace_range])
        first_subspace_index_tensor.append(first_subpace_indices_space[subspace_range])

        sqrt_occupation_numbers_tensor.append(sqrt_space[subspace_range])
        sqrt_first_occupation_numbers_tensor.append(
            sqrt_first_occupation_numbers[subspace_range]
        )

    return {
        "subspace_index_tensor": subspace_index_tensor,
        "first_nonzero_index_tensor": first_nonzero_index_tensor,
        "first_subspace_index_tensor": first_subspace_index_tensor,
        "sqrt_occupation_numbers_tensor": sqrt_occupation_numbers_tensor,
        "sqrt_first_occupation_numbers_tensor": sqrt_first_occupation_numbers_tensor,
    }


def calculate_interferometer_on_fock_space(interferometer, cutoff, calculator):
    """Calculates finite representation of interferometer in the Fock space.
    The function assumes the knowledge of the 1-particle unitary.

    Sources:
    - Fast optimization of parametrized quantum optical circuits
    (https://quantum-journal.org/papers/q-2020-11-30-366/)

    Args:
        interferometer (numpy.ndarray): The 1-particle unitary
        space (np.ndarray): Array of basis elements on the Fock space.

    Returns:
        numpy.ndarray: Finite representation of interferometer in the Fock space
    """

    np = calculator.forward_pass_np
    fallback_np = calculator.fallback_np

    d = interferometer.shape[0]

    indices = np.array([cutoff_cardinality(cutoff=n + 1, d=d) for n in range(cutoff)])

    space = get_fock_space_basis(d=d, cutoff=cutoff)

    identity = fallback_np.identity(d, dtype=int)

    repeated = np.array(
        get_index_in_fock_subspace_array(
            fallback_np.repeat(space[:, None, :], d, axis=1)
            - fallback_np.repeat(identity[None, :, :], len(space), axis=0)
        )
    )

    sqrt_space = np.sqrt(space)
    first_nonzero_space_index = (space != 0).argmax(axis=1)

    space[fallback_np.arange(len(space)), first_nonzero_space_index] -= 1
    first_subpace_indices_space = np.array(get_index_in_fock_subspace_array(space))
    space[fallback_np.arange(len(space)), first_nonzero_space_index] += 1

    sqrt_first_occupation_numbers_space = np.sqrt(
        space[fallback_np.arange(len(space)), first_nonzero_space_index]
    )

    first_nonzero_space_index_array = np.array(first_nonzero_space_index)

    subspace_representations = calculator.accumulator(
        dtype=interferometer.dtype,
        size=cutoff,
        infer_shape=False,
        clear_after_read=False,
    )

    subspace_representations = calculator.write(
        subspace_representations, 0, np.identity(1, interferometer.dtype)
    )

    subspace_representations = calculator.write(
        subspace_representations, 1, interferometer
    )

    for n in calculator.range(2, cutoff):
        subspace_indices = np.mod(
            repeated[indices[n - 1] : indices[n]],
            indices[n - 1] - indices[n - 2],
        )

        first_nonzero_indices = first_nonzero_space_index_array[
            indices[n - 1] : indices[n]
        ]
        first_subspace_indices = first_subpace_indices_space[
            indices[n - 1] : indices[n]
        ]

        sqrt_occupation_numbers = sqrt_space[indices[n - 1] : indices[n]]
        sqrt_first_occupation_numbers = sqrt_first_occupation_numbers_space[
            indices[n - 1] : indices[n]
        ]

        first_part_partially_indexed = interferometer[first_nonzero_indices]

        previous_representation = calculator.read(subspace_representations, n - 1)

        second_part = previous_representation[first_subspace_indices][
            :, subspace_indices
        ]

        matrix = np.einsum(
            "ij,kj,kij->ik",
            sqrt_occupation_numbers,
            first_part_partially_indexed,
            second_part,
        )

        representation = np.transpose(matrix / sqrt_first_occupation_numbers).astype(
            interferometer.dtype
        )

        subspace_representations = calculator.write(
            subspace_representations,
            n,
            representation,
        )

    return subspace_representations


@lru_cache(maxsize=None)
def calculate_index_list_for_appling_interferometer(
    modes: Tuple[int, ...],
    d: int,
    cutoff: int,
) -> List[np.ndarray]:
    subspace = get_fock_space_basis(d=len(modes), cutoff=cutoff)
    auxiliary_subspace = get_fock_space_basis(d=d - len(modes), cutoff=cutoff)

    indices = [cutoff_cardinality(cutoff=n, d=len(modes)) for n in range(cutoff + 1)]
    auxiliary_indices = [
        cutoff_cardinality(cutoff=n, d=d - len(modes)) for n in range(cutoff + 1)
    ]

    auxiliary_modes = get_auxiliary_modes(d, modes)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff - n]

        basis_tensor = np.empty(shape=(size, auxiliary_size, d), dtype=int)

        basis_tensor[:, :, auxiliary_modes] = auxiliary_subspace[:auxiliary_size]

        basis_tensor[:, :, modes] = np.repeat(
            n_particle_subspace[:, None, :], auxiliary_size, axis=1
        )

        index_list.append(get_index_in_fock_space_array(basis_tensor))

    return index_list


def get_projection_operator_indices(d, cutoff, modes, basis_vector):
    new_cutoff = cutoff - np.sum(basis_vector)

    boxes = d - len(modes)

    c = cutoff_cardinality(cutoff=new_cutoff, d=boxes)

    basis = np.empty(shape=(c, d), dtype=int)

    basis[:, modes] = basis_vector

    if len(modes) < d:
        auxiliary_modes = get_auxiliary_modes(d, modes)

        auxiliary_subspace = get_fock_space_basis(d=boxes, cutoff=new_cutoff)

        basis[:, auxiliary_modes] = auxiliary_subspace

    return get_index_in_fock_space_array(basis)
