#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class PNCFockCircuit(BaseFockCircuit):
    def _number(self, operation):
        self.state._add_occupation_number_basis(
            ket=operation.params[0],
            bra=operation.params[1],
            coefficient=operation.params[-1],
        )
