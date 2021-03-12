#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class PureFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._passive_linear,
            "B": self._passive_linear,
            "R": self._passive_linear,
            "MZ": self._passive_linear,
            "F": self._passive_linear,
            "K": self._kerr,
            "CK": self._cross_kerr,
            "MeasureParticleNumber": self._measure_particle_number,
            "Number": self._number,
            "Create": self._create,
            "Annihilate": self._annihilate,
        }

    def _passive_linear(self, operation):
        self.state._apply_passive_linear(
            operator=operation._passive_representation,
            modes=operation.modes
        )

    def _measure_particle_number(self, operation):
        outcome = self.state._measure_particle_number(
            modes=operation.modes,
        )

        # TODO: Better way of providing results
        self.program.results.append(
            Result(measurement=operation, outcome=outcome)
        )

    def _number(self, operation):

        self.state._add_occupation_number_basis(
            occupation_numbers=operation.params[0],
            coefficient=operation.params[1],
            modes=operation.modes,
        )

    def _create(self, operation):
        self.state._apply_creation_operator(operation.modes)

    def _annihilate(self, operation):
        self.state._apply_annihilation_operator(operation.modes)

    def _kerr(self, operation):
        self.state._apply_kerr(
            xi=operation.params[0],
            mode=operation.modes[0],
        )

    def _cross_kerr(self, operation):
        self.state._apply_cross_kerr(
            xi=operation.params[0],
            modes=operation.modes,
        )