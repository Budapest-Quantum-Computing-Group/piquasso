#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.circuit import Circuit

from piquasso import operations


class PNCFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            operations.PassiveTransform.__name__: self._apply,
            operations.B.__name__: self._apply,
            operations.R.__name__: self._apply,
        }

    def _apply(self, operation):
        self.state._apply(
            operator=operation._passive_representation,
            modes=operation.modes
        )
