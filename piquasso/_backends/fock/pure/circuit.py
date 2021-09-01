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

import typing

from piquasso.api.instruction import Instruction
from ..circuit import BaseFockCircuit

if typing.TYPE_CHECKING:
    from .state import PureFockState


class PureFockCircuit(BaseFockCircuit):
    instruction_map = {
        "StateVector": "_state_vector",
        **BaseFockCircuit.instruction_map
    }

    def _state_vector(self, instruction: Instruction, state: "PureFockState") -> None:
        state._add_occupation_number_basis(
            **instruction._all_params,
            modes=instruction.modes,
        )
