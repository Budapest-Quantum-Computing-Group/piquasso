#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class BaseFockCircuit(Circuit, abc.ABC):

    def get_instruction_map(self):
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

    def _passive_linear(self, instruction):
        self.state._apply_passive_linear(
            operator=instruction._passive_representation,
            modes=instruction.modes
        )

    def _measure_particle_number(self, instruction):
        samples = self.state._measure_particle_number(
            modes=instruction.modes,
            shots=instruction.params["shots"],
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _vacuum(self, instruction):
        self.state._apply_vacuum()

    def _create(self, instruction):
        self.state._apply_creation_operator(instruction.modes)

    def _annihilate(self, instruction):
        self.state._apply_annihilation_operator(instruction.modes)

    def _kerr(self, instruction):
        self.state._apply_kerr(
            **instruction.params,
            mode=instruction.modes[0],
        )

    def _cross_kerr(self, instruction):
        self.state._apply_cross_kerr(
            **instruction.params,
            modes=instruction.modes,
        )
