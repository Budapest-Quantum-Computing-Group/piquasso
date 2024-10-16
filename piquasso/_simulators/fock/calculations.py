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
import numba as nb

from scipy.special import comb

from functools import lru_cache

from .state import BaseFockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result
from piquasso.api.exceptions import InvalidInstruction, InvalidParameter

from piquasso._math.fock import (
    cutoff_fock_space_dim,
    cutoff_fock_space_dim_array,
    operator_basis,
    get_fock_space_basis,
    nb_get_fock_space_basis,
)
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_auxiliary_modes,
    get_index_in_fock_space_array,
    get_index_in_fock_subspace,
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

    if state._config.validate and len(modes) != 1:
        raise InvalidInstruction(
            f"The instruction should be specified for '{len(modes)}' "
            f"modes: instruction={instruction}"
        )

    mean_thermal_excitation = instruction._all_params["mean_thermal_excitation"]

    if state._config.validate and not np.isclose(mean_thermal_excitation, 0.0):
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


@nb.njit(cache=True)
def nb_calculate_state_index_matrix_list(d, cutoff, mode):
    auxiliary_subspace = nb_get_fock_space_basis(d=d - 1, cutoff=cutoff)

    auxiliary_modes = get_auxiliary_modes(d, (mode,))
    all_occupation_numbers = np.zeros(d, dtype=np.int32)

    indices = cutoff_fock_space_dim_array(cutoff=np.arange(cutoff + 1), d=d - 1)

    state_index_matrix_list = []

    for n in range(cutoff):
        limit = cutoff - n
        subspace_size = indices[n + 1] - indices[n]

        state_index_matrix = np.empty(shape=(limit, subspace_size), dtype=np.int32)

        for i, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[indices[n] : indices[n + 1]]
        ):
            all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers

            for j in range(limit):
                all_occupation_numbers[mode] = j
                index = get_index_in_fock_space(all_occupation_numbers)
                state_index_matrix[j, i] = index

        state_index_matrix_list.append(state_index_matrix)

    return state_index_matrix_list


calculate_state_index_matrix_list = lru_cache(maxsize=None)(
    nb_calculate_state_index_matrix_list
)


@nb.njit(cache=True)
def calculate_interferometer_helper_indices(d, cutoff):
    space = nb_get_fock_space_basis(d=d, cutoff=cutoff)

    basis = np.empty((space.shape[0], d), dtype=space.dtype)
    first_subpace_indices_space = np.empty(shape=space.shape[0], dtype=space.dtype)
    sqrt_first_occupation_numbers = np.empty(len(space), dtype=np.float64)

    first_nonzero_space_index = np.empty(shape=space.shape[0], dtype=space.dtype)

    sqrt_space = np.empty(shape=space.shape, dtype=np.float64)

    for i in range(len(space)):
        current_basis = space[i]
        sqrt_space[i] = np.sqrt(current_basis)
        found_first = False
        for j in range(d):
            current_basis[j] -= 1
            basis[i, j] = get_index_in_fock_subspace(current_basis)

            if not found_first and current_basis[j] >= 0:
                first_nonzero_space_index[i] = j
                first_subpace_indices_space[i] = basis[i, j]
                found_first = True
                sqrt_first_occupation_numbers[i] = sqrt_space[i, j]

            current_basis[j] += 1

    subspace_index_tensor = []

    first_subspace_index_tensor = []
    first_nonzero_index_tensor = []

    sqrt_occupation_numbers_tensor = []
    sqrt_first_occupation_numbers_tensor = []

    indices = cutoff_fock_space_dim_array(cutoff=np.arange(1, cutoff + 1), d=d)

    for n in range(2, cutoff):
        subspace_range = np.arange(indices[n - 1], indices[n])
        subspace_index_tensor.append(
            np.mod(basis[subspace_range], indices[n - 1] - indices[n - 2])
        )

        first_nonzero_index_tensor.append(first_nonzero_space_index[subspace_range])
        first_subspace_index_tensor.append(first_subpace_indices_space[subspace_range])

        sqrt_occupation_numbers_tensor.append(sqrt_space[subspace_range])
        sqrt_first_occupation_numbers_tensor.append(
            sqrt_first_occupation_numbers[subspace_range]
        )

    return (
        subspace_index_tensor,
        first_nonzero_index_tensor,
        first_subspace_index_tensor,
        sqrt_occupation_numbers_tensor,
        sqrt_first_occupation_numbers_tensor,
    )


@nb.njit(cache=True)
def nb_calculate_index_list_for_appling_interferometer(
    modes: Tuple[int, ...],
    d: int,
    cutoff: int,
) -> List[np.ndarray]:
    subspace = nb_get_fock_space_basis(d=len(modes), cutoff=cutoff)
    auxiliary_subspace = nb_get_fock_space_basis(d=d - len(modes), cutoff=cutoff)

    indices = cutoff_fock_space_dim_array(cutoff=np.arange(cutoff + 1), d=len(modes))
    auxiliary_indices = cutoff_fock_space_dim_array(
        cutoff=np.arange(cutoff + 1), d=d - len(modes)
    )
    auxiliary_modes = get_auxiliary_modes(d, modes)

    all_occupation_numbers = np.zeros(d, dtype=np.int32)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff - n]
        state_index_matrix = np.empty(shape=(size, auxiliary_size), dtype=np.int32)
        for idx1, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[:auxiliary_size]
        ):
            for idx, mode in enumerate(auxiliary_modes):
                all_occupation_numbers[mode] = auxiliary_occupation_numbers[idx]

            for idx2, column_vector_on_subspace in enumerate(n_particle_subspace):
                for idx, mode in enumerate(modes):
                    all_occupation_numbers[mode] = column_vector_on_subspace[idx]

                column_index = get_index_in_fock_space(all_occupation_numbers)
                state_index_matrix[idx2, idx1] = column_index

        index_list.append(state_index_matrix)

    return index_list


calculate_index_list_for_appling_interferometer = lru_cache(maxsize=None)(
    nb_calculate_index_list_for_appling_interferometer
)


def get_projection_operator_indices(d, cutoff, modes, basis_vector):
    new_cutoff = cutoff - np.sum(basis_vector)

    boxes = d - len(modes)

    card = cutoff_fock_space_dim(cutoff=new_cutoff, d=boxes)

    basis = np.empty(shape=(card, d), dtype=int)

    basis[:, modes] = basis_vector

    if len(modes) < d:
        auxiliary_modes = get_auxiliary_modes(d, modes)

        auxiliary_subspace = get_fock_space_basis(d=boxes, cutoff=new_cutoff)

        basis[:, auxiliary_modes] = auxiliary_subspace

    return get_index_in_fock_space_array(basis)
