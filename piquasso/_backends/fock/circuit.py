#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            "Squeezing": self._linear,
            "Displacement": self._linear,
            "Squeezing2": self._linear,
            "GaussianTransform": self._linear,
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

    def _linear(self, instruction):
        self.state._apply_linear(
            passive_representation=instruction._passive_representation,
            active_representation=instruction._active_representation,
            displacement=instruction._displacement_vector,
            modes=instruction.modes,
        )
