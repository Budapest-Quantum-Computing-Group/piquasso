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

import numpy as np

from piquasso.api.circuit import Circuit
from piquasso.api.result import Result


class GaussianCircuit(Circuit):

    def get_instruction_map(self):
        return {
            "Interferometer": self._passive_linear,
            "Beamsplitter": self._passive_linear,
            "Phaseshifter": self._passive_linear,
            "MachZehnder": self._passive_linear,
            "Fourier": self._passive_linear,
            "GaussianTransform": self._linear,
            "Squeezing": self._linear,
            "QuadraticPhase": self._linear,
            "Squeezing2": self._linear,
            "ControlledX": self._linear,
            "ControlledZ": self._linear,
            "Displacement": self._displacement,
            "PositionDisplacement": self._displacement,
            "MomentumDisplacement": self._displacement,
            "Graph": self._graph,
            "HomodyneMeasurement": self._homodyne_measurement,
            "HeterodyneMeasurement": self._generaldyne_measurement,
            "GeneraldyneMeasurement": self._generaldyne_measurement,
            "Vacuum": self._vacuum,
            "Mean": self._mean,
            "Covariance": self._covariance,
            "ParticleNumberMeasurement": self._particle_number_measurement,
            "ThresholdMeasurement": self._threshold_measurement,
        }

    def _passive_linear(self, instruction):
        self.state._apply_passive_linear(
            instruction._passive_block,
            instruction.modes
        )

    def _linear(self, instruction):
        self.state._apply_linear(
            passive_block=instruction._passive_block,
            active_block=instruction._active_block,
            modes=instruction.modes
        )

    def _displacement(self, instruction):
        self.state._apply_displacement(
            displacement_vector=instruction._displacement_vector,
            modes=instruction.modes,
        )

    def _homodyne_measurement(self, instruction):
        phi = instruction.params["phi"]
        modes = instruction.modes

        phaseshift = np.identity(len(modes)) * np.exp(- 1j * phi)

        self.state._apply_passive_linear(
            phaseshift,
            modes=modes,
        )

        samples = self.state._apply_generaldyne_measurement(
            detection_covariance=instruction.params["detection_covariance"],
            shots=instruction.params["shots"],
            modes=modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _generaldyne_measurement(self, instruction):
        samples = self.state._apply_generaldyne_measurement(
            detection_covariance=instruction.params["detection_covariance"],
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _vacuum(self, instruction):
        self.state.reset()

    def _mean(self, instruction):
        self.state.mean = instruction.params["mean"]

    def _covariance(self, instruction):
        self.state.cov = instruction.params["cov"]

    def _particle_number_measurement(self, instruction):
        samples = self.state._apply_particle_number_measurement(
            cutoff=instruction.params["cutoff"],
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _threshold_measurement(self, instruction):
        samples = self.state._apply_threshold_measurement(
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _graph(self, instruction):
        """
        TODO: Find a better solution for multiple operations.
        """
        instruction._squeezing.modes = instruction.modes
        instruction._interferometer.modes = instruction.modes

        self._linear(instruction._squeezing)
        self._passive_linear(instruction._interferometer)
