#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class PureFockCircuit(BaseFockCircuit):
    def _number(self, operation):
        self.state._add_occupation_number_basis(
            occupation_numbers=operation.params[0],
            coefficient=operation.params[-1],
            modes=operation.modes,
        )
