#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class PNCFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply,
            "B": self._apply,
            "R": self._apply,
            "MZ": self._apply,
            "F": self._apply,
            "MeasureParticleNumber": self._measure_particle_number,
            # TODO: Cross-Kerr gate.
        }

    def _apply(self, operation):
        self.state._apply(
            operator=operation._passive_representation,
            modes=operation.modes
        )

    def _measure_particle_number(self, operation):
        outcome = self.state._measure_particle_number()

        # TODO: Better way of providing results
        self.program.results.append(
            Result(measurement=operation, outcome=outcome)
        )
