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
from BoSS.BosonSamplingSimulator import BosonSamplingSimulator
from BoSS.simulation_strategies.GeneralizedCliffordsSimulationStrategy \
    import GeneralizedCliffordsSimulationStrategy

from piquasso.api.circuit import Circuit


class SamplingCircuit(Circuit):
    r"""Circuit for Boson Sampling."""

    def get_instruction_map(self):
        return {
            "Beamsplitter": self._passive_linear,
            "Phaseshifter": self._passive_linear,
            "MachZehnder": self._passive_linear,
            "Fourier": self._passive_linear,
            "Sampling": self._sampling,
            "Interferometer": self._passive_linear,
            "Loss": self._loss,
        }

    def _passive_linear(self, instruction):
        r"""Applies an interferometer to the circuit.

        This can be interpreted as placing another interferometer in the network, just
        before performing the sampling. This instruction is realized by multiplying
        current effective interferometer matrix with new interferometer matrix.

        Do note, that new interferometer matrix works as interferometer matrix on
        qumodes (provided as the arguments) and as an identity on every other mode.
        """

        self.state._apply_passive_linear(
            instruction._passive_block,
            instruction.modes,
        )

    def _sampling(self, instruction):
        simulation_strategy = GeneralizedCliffordsSimulationStrategy(
            self.state.interferometer
        )
        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        initial_state = np.array(self.state.initial_state)
        self.state.results = sampling_simulator.get_classical_simulation_results(
            initial_state,
            samples_number=instruction.params["shots"]
        )

    def _loss(self, instruction):
        self.state._apply_loss(
            transmissivity=instruction._transmissivity,
            modes=instruction.modes,
        )
