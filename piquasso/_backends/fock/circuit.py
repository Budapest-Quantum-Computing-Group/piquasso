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
import typing

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit
from piquasso.api.instruction import Instruction

if typing.TYPE_CHECKING:
    from .state import BaseFockState


class BaseFockCircuit(Circuit, abc.ABC):

    instruction_map = {
        "Interferometer": "_passive_linear",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "Kerr": "_kerr",
        "CrossKerr": "_cross_kerr",
        "Squeezing": "_linear",
        "QuadraticPhase": "_linear",
        "Displacement": "_linear",
        "PositionDisplacement": "_linear",
        "MomentumDisplacement": "_linear",
        "Squeezing2": "_linear",
        "GaussianTransform": "_linear",
        "ParticleNumberMeasurement": "_particle_number_measurement",
        "Vacuum": "_vacuum",
        "Create": "_create",
        "Annihilate": "_annihilate",
    }

    def _passive_linear(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_passive_linear(
            operator=instruction._all_params["passive_block"],
            modes=instruction.modes
        )

    def _particle_number_measurement(
        self, instruction: Instruction, state: "BaseFockState"
    ) -> None:
        samples = state._particle_number_measurement(
            modes=instruction.modes,
            shots=self.shots,
        )

        self.result = Result(instruction=instruction, samples=samples)  # type: ignore

    def _vacuum(self, _instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_vacuum()

    def _create(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_creation_operator(instruction.modes)

    def _annihilate(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_annihilation_operator(instruction.modes)

    def _kerr(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_kerr(
            **instruction._all_params,
            mode=instruction.modes[0],
        )

    def _cross_kerr(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_cross_kerr(
            **instruction._all_params,
            modes=instruction.modes,  # type: ignore
        )

    def _linear(self, instruction: Instruction, state: "BaseFockState") -> None:
        state._apply_linear(
            passive_block=instruction._all_params["passive_block"],
            active_block=instruction._all_params["active_block"],
            displacement=instruction._all_params["displacement_vector"],
            modes=instruction.modes,
        )
