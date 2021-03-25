#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class PureFockCircuit(BaseFockCircuit):
    def get_instruction_map(self):
        return {
            "StateVector": self._state_vector,
            **super().get_instruction_map()
        }

    def _state_vector(self, instruction):
        self.state._add_occupation_number_basis(
            **instruction.params,
            modes=instruction.modes,
        )
