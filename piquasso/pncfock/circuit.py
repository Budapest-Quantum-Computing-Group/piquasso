#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.circuit import Circuit


class PNCFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply,
            "B": self._apply,
            "R": self._apply,
            "MZ": self._apply,
            "F": self._apply,
        }

    def _apply(self, operation):
        self.state._apply(
            operator=operation._passive_representation,
            modes=operation.modes
        )
