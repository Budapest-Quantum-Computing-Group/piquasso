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

from piquasso._math.fock import cutoff_cardinality, FockSpace
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_auxiliary_modes,
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

    space = state._space

    new_state = state._as_mixed()

    new_density_matrix = new_state._get_empty()

    for index, basis in space.operator_basis:
        coefficient = new_state._density_matrix[index]

        ket = np.asarray(basis.ket)
        bra = np.asarray(basis.bra)

        n = ket[modes]
        m = bra[modes]

        common_term = coefficient * np.cos(theta) ** (n + m)

        for k in range(min(n, m) + 1):
            current_ket = np.copy(ket)
            current_ket[(modes,)] -= k

            current_bra = np.copy(bra)
            current_bra[(modes,)] -= k

            current_ket_index = space.index(tuple(current_ket))
            current_bra_index = space.index(tuple(current_bra))

            current_index = (current_ket_index, current_bra_index)

            new_density_matrix[current_index] += common_term * (
                np.tan(theta) ** (2 * k) * np.sqrt(comb(n, k) * comb(m, k))
            )

    new_state._density_matrix = new_density_matrix

    return Result(state=new_state)


@lru_cache(maxsize=None)
def calculate_state_index_matrix_list(space, auxiliary_subspace, mode):
    d = space.d
    cutoff = space.cutoff

    state_index_matrix_list = []
    auxiliary_modes = get_auxiliary_modes(d, (mode,))
    all_occupation_numbers = np.zeros(d, dtype=int)

    indices = [cutoff_cardinality(cutoff=n - 1, d=d - 1) for n in range(1, cutoff + 2)]

    for n in range(cutoff):
        limit = space.cutoff - n
        subspace_size = indices[n + 1] - indices[n]

        state_index_matrix = np.empty(shape=(limit, subspace_size), dtype=int)

        for i, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[indices[n] : indices[n + 1]]
        ):
            all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers

            for j in range(limit):
                all_occupation_numbers[mode] = j
                state_index_matrix[j, i] = get_index_in_fock_space(
                    tuple(all_occupation_numbers)
                )

        state_index_matrix_list.append(state_index_matrix)

    return state_index_matrix_list


@lru_cache(maxsize=None)
def calculate_interferometer_helper_indices(space):
    d = space.d
    cutoff = space.cutoff
    space = [np.array(element, dtype=int) for element in space]
    indices = [cutoff_cardinality(cutoff=n - 1, d=d) for n in range(2, cutoff + 2)]

    subspace_index_tensor = []
    first_subspace_index_tensor = []

    nonzero_index_tensor = []
    first_nonzero_index_tensor = []

    sqrt_occupation_numbers_tensor = []
    first_occupation_numbers_tensor = []

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]
        subspace = space[indices[n - 1] : indices[n]]

        subspace_indices = []
        first_subspace_indices = []

        nonzero_indices = []
        first_nonzero_indices = []

        sqrt_occupation_numbers = []
        first_occupation_numbers = np.empty(size)

        for index, vector in enumerate(subspace):
            nonzero_multiindex = np.nonzero(vector)[0]
            first_nonzero_multiindex = nonzero_multiindex[0]

            subspace_multiindex = []
            for nonzero_index in nonzero_multiindex:
                vector[nonzero_index] -= 1
                subspace_multiindex.append(get_index_in_fock_subspace(tuple(vector)))
                vector[nonzero_index] += 1

            subspace_indices.append(subspace_multiindex)
            first_subspace_indices.append(subspace_multiindex[0])

            nonzero_indices.append(nonzero_multiindex)
            first_nonzero_indices.append(first_nonzero_multiindex)

            sqrt_occupation_numbers.append(np.sqrt(vector[nonzero_multiindex]))
            first_occupation_numbers[index] = vector[first_nonzero_multiindex]

        subspace_index_tensor.append(subspace_indices)
        first_nonzero_index_tensor.append(first_nonzero_indices)

        nonzero_index_tensor.append(nonzero_indices)
        first_subspace_index_tensor.append(first_subspace_indices)

        sqrt_occupation_numbers_tensor.append(sqrt_occupation_numbers)
        first_occupation_numbers_tensor.append(first_occupation_numbers)

    return {
        "subspace_index_tensor": subspace_index_tensor,
        "first_nonzero_index_tensor": first_nonzero_index_tensor,
        "nonzero_index_tensor": nonzero_index_tensor,
        "first_subspace_index_tensor": first_subspace_index_tensor,
        "sqrt_occupation_numbers_tensor": sqrt_occupation_numbers_tensor,
        "first_occupation_numbers_tensor": first_occupation_numbers_tensor,
        "indices": indices,
    }


def calculate_interferometer_on_fock_space(interferometer, index_dict, calculator):
    """Calculates finite representation of interferometer in the Fock space.
    The function assumes the knowledge of the 1-particle unitary.

    Sources:
    - Fast optimization of parametrized quantum optical circuits
    (https://quantum-journal.org/papers/q-2020-11-30-366/)

    Args:
        interferometer (numpy.ndarray): The 1-particle unitary
        space (FockSpace): List of basis elements on the Fock space.

    Returns:
        numpy.ndarray: Finite representation of interferometer in the Fock space
    """

    np = calculator.forward_pass_np

    subspace_representations = []

    subspace_representations.append(np.array([[1.0]], dtype=interferometer.dtype))
    subspace_representations.append(interferometer)

    indices = index_dict["indices"]
    cutoff = len(indices)

    for n in range(2, cutoff):
        size = indices[n] - indices[n - 1]

        subspace_indices = index_dict["subspace_index_tensor"][n - 2]
        first_subspace_indices = index_dict["first_subspace_index_tensor"][n - 2]

        nonzero_indices = index_dict["nonzero_index_tensor"][n - 2]
        first_nonzero_indices = index_dict["first_nonzero_index_tensor"][n - 2]

        sqrt_occupation_numbers = index_dict["sqrt_occupation_numbers_tensor"][n - 2]
        first_occupation_numbers = index_dict["first_occupation_numbers_tensor"][n - 2]

        first_part_partially_indexed = interferometer[first_nonzero_indices, :]
        second_part_partially_indexed = subspace_representations[n - 1][
            first_subspace_indices, :
        ]

        matrix = []

        for index in range(size):
            first_part = (
                sqrt_occupation_numbers[index]
                * first_part_partially_indexed[:, nonzero_indices[index]]
            )
            second_part = second_part_partially_indexed[:, subspace_indices[index]]
            matrix.append(np.einsum("ij,ij->i", first_part, second_part))

        new_subspace_representation = np.transpose(
            np.array(matrix) / np.sqrt(first_occupation_numbers)
        )

        subspace_representations.append(
            new_subspace_representation.astype(interferometer.dtype)
        )

    return subspace_representations


@lru_cache(maxsize=None)
def calculate_index_list_for_appling_interferometer(
    modes: Tuple[int, ...],
    space: FockSpace,
) -> List[np.ndarray]:
    cutoff = space.cutoff
    d = space.d
    calculator = space._calculator

    subspace = FockSpace(
        d=len(modes), cutoff=space.cutoff, calculator=calculator, config=space.config
    )
    auxiliary_subspace = FockSpace(
        d=d - len(modes),
        cutoff=space.cutoff,
        calculator=calculator,
        config=space.config,
    )

    indices = [cutoff_cardinality(cutoff=n, d=len(modes)) for n in range(cutoff + 1)]
    auxiliary_indices = [
        cutoff_cardinality(cutoff=n, d=d - len(modes)) for n in range(cutoff + 1)
    ]

    auxiliary_modes = get_auxiliary_modes(d, modes)
    all_occupation_numbers = np.zeros(d, dtype=int)

    index_list = []

    for n in range(cutoff):
        size = indices[n + 1] - indices[n]
        n_particle_subspace = subspace[indices[n] : indices[n + 1]]
        auxiliary_size = auxiliary_indices[cutoff - n]
        state_index_matrix = np.empty(shape=(size, auxiliary_size), dtype=int)
        for idx1, auxiliary_occupation_numbers in enumerate(
            auxiliary_subspace[:auxiliary_size]
        ):
            all_occupation_numbers[(auxiliary_modes,)] = auxiliary_occupation_numbers
            for idx2, column_vector_on_subspace in enumerate(n_particle_subspace):
                all_occupation_numbers[(modes,)] = column_vector_on_subspace
                column_index = get_index_in_fock_space(tuple(all_occupation_numbers))
                state_index_matrix[idx2, idx1] = column_index

        index_list.append(state_index_matrix)

    return index_list
