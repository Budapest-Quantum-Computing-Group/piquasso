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

import numpy as np

from scipy.special import comb

from .state import BaseFockState

from piquasso.api.instruction import Instruction
from piquasso.api.result import Result
from piquasso.api.exceptions import InvalidInstruction, InvalidParameter


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
