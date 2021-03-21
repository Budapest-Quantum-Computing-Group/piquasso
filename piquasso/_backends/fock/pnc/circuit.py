#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class PNCFockCircuit(BaseFockCircuit):
    def get_operation_map(self):
        return {
            "DensityMatrix": self._density_matrix,
            **super().get_operation_map()
        }

    def _density_matrix(self, operation):
        self.state._add_occupation_number_basis(
            ket=operation.params[0],
            bra=operation.params[1],
            coefficient=operation.params[-1],
        )
