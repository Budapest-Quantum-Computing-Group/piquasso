#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class BaseFockCircuit(Circuit, abc.ABC):

    def get_operation_map(self):
        return {
            "Interferometer": self._passive_linear,
            "Beamsplitter": self._passive_linear,
            "Phaseshifter": self._passive_linear,
            "MachZehnder": self._passive_linear,
            "Fourier": self._passive_linear,
            "Kerr": self._kerr,
            "CrossKerr": self._cross_kerr,
            "MeasureParticleNumber": self._measure_particle_number,
            "Vacuum": self._vacuum,
            "Create": self._create,
            "Annihilate": self._annihilate,
        }

    def _passive_linear(self, operation):
        self.state._apply_passive_linear(
            operator=operation._passive_representation,
            modes=operation.modes
        )

    def _measure_particle_number(self, operation):
        outcomes = self.state._measure_particle_number(
            modes=operation.modes,
            shots=operation.params["shots"],
        )

        self._add_result(
            [
                Result(operation=operation, outcome=outcome)
                for outcome in outcomes
            ]
        )

    def _vacuum(self, operation):
        self.state._apply_vacuum()

    def _create(self, operation):
        self.state._apply_creation_operator(operation.modes)

    def _annihilate(self, operation):
        self.state._apply_annihilation_operator(operation.modes)

    def _kerr(self, operation):
        self.state._apply_kerr(
            **operation.params,
            mode=operation.modes[0],
        )

    def _cross_kerr(self, operation):
        self.state._apply_cross_kerr(
            **operation.params,
            modes=operation.modes,
        )
