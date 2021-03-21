#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class PureFockCircuit(BaseFockCircuit):
    def get_operation_map(self):
        return {
            "StateVector": self._state_vector,
            **super().get_operation_map()
        }

    def _state_vector(self, operation):
        self.state._add_occupation_number_basis(
            occupation_numbers=operation.params[0],
            coefficient=operation.params[-1],
            modes=operation.modes,
        )
