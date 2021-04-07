#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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

    def _squeezing(self, instruction):
        warnings.warn(
            f"Squeezing the state with instruction {instruction} may not result in "
            f"the desired state, since state {self.state.__class__} only stores a "
            "limited amount of information to handle particle number conserving "
            "instructions.\n"
            "Consider using 'FockState' or 'PureFockState' instead!",
            UserWarning
        )

        super()._squeezing(instruction)

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
