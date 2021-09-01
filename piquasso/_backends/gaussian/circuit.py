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
import typing

from piquasso.api.circuit import Circuit
from piquasso.api.instruction import Instruction
from piquasso.api.result import Result

if typing.TYPE_CHECKING:
    from . import GaussianState


class GaussianCircuit(Circuit):

    instruction_map = {
        "Interferometer": "_passive_linear",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "GaussianTransform": "_linear",
        "Squeezing": "_linear",
        "QuadraticPhase": "_linear",
        "Squeezing2": "_linear",
        "ControlledX": "_linear",
        "ControlledZ": "_linear",
        "Displacement": "_displacement",
        "PositionDisplacement": "_displacement",
        "MomentumDisplacement": "_displacement",
        "Graph": "_graph",
        "HomodyneMeasurement": "_homodyne_measurement",
        "HeterodyneMeasurement": "_generaldyne_measurement",
        "GeneraldyneMeasurement": "_generaldyne_measurement",
        "Vacuum": "_vacuum",
        "Mean": "_mean",
        "Covariance": "_covariance",
        "ParticleNumberMeasurement": "_particle_number_measurement",
        "ThresholdMeasurement": "_threshold_measurement",
    }

    def _passive_linear(self, instruction: Instruction, state: "GaussianState") -> None:
        state._apply_passive_linear(
            instruction._all_params["passive_block"],
            instruction.modes
        )

    def _linear(self, instruction: Instruction, state: "GaussianState") -> None:
        state._apply_linear(
            passive_block=instruction._all_params["passive_block"],
            active_block=instruction._all_params["active_block"],
            modes=instruction.modes
        )

    def _displacement(self, instruction: Instruction, state: "GaussianState") -> None:
        state._apply_displacement(
            displacement_vector=instruction._all_params["displacement_vector"],
            modes=instruction.modes,
        )

    def _homodyne_measurement(
        self, instruction: Instruction, state: "GaussianState"
    ) -> None:
        phi = instruction._all_params["phi"]
        modes = instruction.modes

        phaseshift = np.identity(len(modes)) * np.exp(- 1j * phi)

        state._apply_passive_linear(
            phaseshift,
            modes=modes,
        )

        samples = state._apply_generaldyne_measurement(
            detection_covariance=instruction._all_params["detection_covariance"],
            shots=self.shots,
            modes=modes,
        )

        self.result = Result(instruction=instruction, samples=samples)

    def _generaldyne_measurement(
        self, instruction: Instruction, state: "GaussianState"
    ) -> None:
        samples = state._apply_generaldyne_measurement(
            detection_covariance=instruction._all_params["detection_covariance"],
            shots=self.shots,
            modes=instruction.modes,
        )

        self.result = Result(instruction=instruction, samples=samples)

    def _vacuum(self, _instruction: Instruction, state: "GaussianState") -> None:
        state.reset()

    def _mean(self, instruction: Instruction, state: "GaussianState") -> None:
        state.xpxp_mean_vector = instruction._all_params["mean"]

    def _covariance(self, instruction: Instruction, state: "GaussianState") -> None:
        state.xpxp_covariance_matrix = instruction._all_params["cov"]

    def _particle_number_measurement(
        self, instruction: Instruction, state: "GaussianState"
    ) -> None:
        samples = state._apply_particle_number_measurement(
            cutoff=instruction._all_params["cutoff"],
            shots=self.shots,
            modes=instruction.modes,
        )

        self.result = Result(instruction=instruction, samples=samples)

    def _threshold_measurement(
        self, instruction: Instruction, state: "GaussianState"
    ) -> None:
        samples = state._apply_threshold_measurement(
            shots=self.shots,
            modes=instruction.modes,
        )

        self.result = Result(instruction=instruction, samples=samples)

    def _graph(self, instruction: Instruction, state: "GaussianState") -> None:
        """
        TODO: Find a better solution for multiple operations.
        """
        instruction._all_params["squeezing"].modes = instruction.modes
        instruction._all_params["interferometer"].modes = instruction.modes

        self._linear(instruction._all_params["squeezing"], state)
        self._passive_linear(instruction._all_params["interferometer"], state)
