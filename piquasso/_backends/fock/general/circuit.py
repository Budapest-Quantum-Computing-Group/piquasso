#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class FockCircuit(BaseFockCircuit):
    def get_instruction_map(self):
        return {
            "DensityMatrix": self._density_matrix,
            **super().get_instruction_map()
        }

    def _density_matrix(self, instruction):
        self.state._add_occupation_number_basis(**instruction.params)
