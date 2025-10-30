#
# Copyright 2021-2025 Budapest Quantum Computing Group
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

import piquasso as pq
import numpy as np

# TODO: need to check that Qiskit is installed if this file is imported

class DualRailConverter:

    @staticmethod
    def bosonic_hadamard(mode1, mode2):
        instructions = []
        instructions.append(pq.Beamsplitter(np.pi / 4).on_modes(mode1, mode2))
        instructions.append(pq.Phaseshifter(np.pi).on_modes(mode2))
        return instructions

    @staticmethod
    def cz_on_two_bosonic_qubits(modes):
        """Note: requires two auxiliary modes with one photon each."""
        
        # Knill's notation
        cz_beamsplitter_first_theta_value = 54.74 / 180 * np.pi
        cz_beamsplitter_second_theta_value = 17.63 / 180 * np.pi

        instructions = []

        instructions.append(pq.Phaseshifter(np.pi).on_modes(modes[0]))
        instructions.append(pq.Phaseshifter(np.pi).on_modes(modes[1]))

        instructions.append(
            pq.Beamsplitter(cz_beamsplitter_first_theta_value).on_modes(modes[0], modes[2])
        )
        instructions.append(
            pq.Beamsplitter(cz_beamsplitter_first_theta_value).on_modes(modes[1], modes[3])
        )
        instructions.append(
            pq.Beamsplitter(-1 * cz_beamsplitter_first_theta_value).on_modes(modes[0], modes[1])
        )
        instructions.append(
            pq.Beamsplitter(cz_beamsplitter_second_theta_value).on_modes(modes[2], modes[3])
        )
        instructions.append(
            pq.PostSelectPhotons(
                photon_counts=[1, 1],
            ).on_modes(modes[2], modes[3])
        )
        return instructions

    def _convert_from_qiskit_qc(self, qc: "QuantumCircuit") -> pq.Program:
    
        # TODO: Validate qc such that it is only MBQC with H and CZ gates (for now)
        num_cz = 0
        for instruction in qc.data:
            if instruction.name not in ["h", "cz", "measure"]:
                raise ValueError(f"Unsupported instruction '{instruction.name}' in the quantum circuit.")
            if instruction.name == "cz":
                num_cz += 1

        num_bosonic_qubits = qc.num_qubits

        # |0> = [0, 1]
        # |1> = [1, 0]
        modes_with_one_photon = [i for i in range(0, num_bosonic_qubits * 2, 2)]
        modes_with_zero_photon = [i for i in range(1, num_bosonic_qubits * 2 + 1, 2)]

        num_aux_needed = num_cz * 2
        aux_modes = [i for i in range(num_bosonic_qubits * 2, num_bosonic_qubits * 2 + num_aux_needed)]
        instructions = []
    
        modes_with_one_photon = modes_with_one_photon + aux_modes
        preparations = [
            pq.Create().on_modes(*modes_with_one_photon),
            pq.Vacuum().on_modes(*modes_with_zero_photon)
        ]
        instructions.extend(preparations)

        cz_idx = 0
        for instruction in qc.data:
            qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
            if instruction.name == "h":
                qubit = qubit_indices[0]
                pq_instruction = self.bosonic_hadamard(qubit, qubit + num_bosonic_qubits)
                instructions.extend(pq_instruction)
            elif instruction.name == "cz":
                mode_indices = [q * 2 for q in qubit_indices]
                aux_modes_this_cz = [aux_modes[cz_idx * 2], aux_modes[cz_idx * 2 + 1]]
                pq_instruction = self.cz_on_two_bosonic_qubits(mode_indices + aux_modes_this_cz)
                instructions.extend(pq_instruction)
                cz_idx += 1
            elif instruction.name == "measure":
                pq_instruction = pq.ParticleNumberMeasurement().on_modes(2*qubit_indices[0], 2*qubit_indices[0] + 1)
                instructions.append(pq_instruction)
        return pq.Program(instructions=instructions)