#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class PureFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply,
            "B": self._apply,
            "R": self._apply,
            "MZ": self._apply,
            "F": self._apply,
            "MeasureParticleNumber": self._measure_particle_number,
            "Number": self._number,
            "Create": self._create,
            "Annihilate": self._annihilate,
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

    def _number(self, operation):
        occupation_numbers = operation.params[0]
        coefficient = operation.params[1]

        self.state._add_occupation_number_basis(coefficient, occupation_numbers)

    def _create(self, operation):
        modes = operation.modes

        self.state._apply_creation_operator(modes)

    def _annihilate(self, operation):
        modes = operation.modes

        self.state._apply_annihilation_operator(modes)
