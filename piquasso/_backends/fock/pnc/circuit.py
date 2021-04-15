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

import warnings

from ..circuit import BaseFockCircuit


class PNCFockCircuit(BaseFockCircuit):
    def get_instruction_map(self):
        return {
            "DensityMatrix": self._density_matrix,
            **super().get_instruction_map()
        }

    def _density_matrix(self, instruction):
        self.state._add_occupation_number_basis(**instruction.params)

    def _linear(self, instruction):
        warnings.warn(
            f"Gaussian evolution of the state with instruction {instruction} may not "
            f"result in the desired state, since state {self.state.__class__} only "
            "stores a limited amount of information to handle particle number "
            "conserving instructions.\n"
            "Consider using 'FockState' or 'PureFockState' instead!",
            UserWarning
        )

        super()._linear(instruction)

    def _displacement(self, instruction):
        warnings.warn(
            f"Displacing the state with instruction {instruction} may not result in "
            f"the desired state, since state {self.state.__class__} only stores a "
            "limited amount of information to handle particle number conserving "
            "instructions.\n"
            "Consider using 'FockState' or 'PureFockState' instead!",
            UserWarning
        )

        super()._displacement(instruction)
