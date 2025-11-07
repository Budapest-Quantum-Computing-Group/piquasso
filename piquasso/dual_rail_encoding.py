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

# Encoding used
# --------------
#
# Dual-rail encoding of a single bosonic qubit into two bosonic modes:
#
# |0> = |0, 1>
# |1> = |1, 0>
#
# where |n, m> denotes n photons in the first mode and m photons in the second mode and
# the left-hand side of each equation corresponds to the bosonic qubit states.
zero_bosonic_qubit_state = [0, 1]
one_bosonic_qubit_state = [1, 0]

def prep_bosonic_qubits(all_modes, modes_with_one_photon) -> list:
    """Prepares a bosonic qubits in the specified basis states.

    The following encoding is being used:

          |0> = [0, 1]
          |1> = [1, 0]

    Args:
        all_modes: All the modes to first prepare in the vacuum state.
        modes_with_one_photon: The modes where a single photon is created.

    Returns:
        A list of Piquasso instructions to prepare the bosonic qubit.
    """
    instructions = []
    if all_modes:
        instructions.append(pq.Vacuum().on_modes(*all_modes))
    if modes_with_one_photon:
        instructions.append(pq.Create().on_modes(*modes_with_one_photon))
    return instructions

def paulix_bosonic(mode1, mode2):
    """Applies a Pauli-X gate on a bosonic qubit encoded in dual-rail format.

    Args:
        mode: The mode of the bosonic qubit.
    """
    instructions = []
    instructions.append(pq.Beamsplitter(np.pi / 2).on_modes(mode1, mode2))
    instructions.append(pq.Phaseshifter(np.pi).on_modes(mode1))
    return instructions

def pauliz_bosonic(mode1, mode2):
    """Applies a Pauli-Z gate on a bosonic qubit encoded in dual-rail format.

    Args:
        mode: The mode of the bosonic qubit.
    """
    return phase_gate_bosonic(np.pi, mode1)

def phase_gate_bosonic(theta, mode):
    """Applies a phase gate on a bosonic qubit encoded in dual-rail format.

    Args:
        theta: The phase angle.
        mode: The first mode of the bosonic qubit (the second mode is mode + 1).
    Returns:
        A list of Piquasso instructions implementing the phase gate.
    """
    instructions = []
    if not np.isclose(theta, 0.0):
        instructions.append(pq.Phaseshifter(theta).on_modes(mode))
    return instructions

def hadamard_bosonic(mode1, mode2):
    instructions = []
    instructions.append(pq.Beamsplitter(np.pi / 4).on_modes(mode1, mode2))
    instructions.append(pq.Phaseshifter(np.pi).on_modes(mode1))
    return instructions

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

def _encode_dual_rail_from_qiskit(qc: "QuantumCircuit") ->  pq.Program:

    supported_instructions = {"h", "cz", "p", "measure"}

    num_cz = 0
    for instruction in qc.data:
        if instruction.name not in supported_instructions:
            raise ValueError(f"Unsupported instruction '{instruction.name}' in the quantum circuit.")
        if instruction.name == "cz":
            num_cz += 1

    num_bosonic_qubits = qc.num_qubits

    # |0> = [0, 1]
    # |1> = [1, 0]
    modes_with_one_photon = list(range(0, num_bosonic_qubits * 2, 2))
    num_aux_needed = num_cz * 2
    aux_modes = list(range(num_bosonic_qubits * 2, num_bosonic_qubits * 2 + num_aux_needed))

    all_modes = list(range(num_bosonic_qubits * 2)) + aux_modes

    instructions = []

    modes_with_one_photon = modes_with_one_photon + aux_modes

    preparations = prep_bosonic_qubits(all_modes, modes_with_one_photon)

    instructions.extend(preparations)

    cz_idx = 0
    for instruction in qc.data:
        qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
        if instruction.name == "h":
            qubit = qubit_indices[0]
            pq_instruction = hadamard_bosonic(2 * qubit, 2 * qubit + 1)
            instructions.extend(pq_instruction)
        elif instruction.name == "x":
            qubit = qubit_indices[0]
            pq_instruction = paulix_bosonic(2 * qubit, 2 * qubit + 1)
            instructions.extend(pq_instruction)
        elif instruction.name == "z":
            qubit = qubit_indices[0]
            pq_instruction = pauliz_bosonic(2 * qubit, 2 * qubit + 1)
            instructions.extend(pq_instruction)
        elif instruction.name == "cz":
            mode_indices = [q * 2 for q in qubit_indices]
            aux_modes_this_cz = [aux_modes[cz_idx * 2], aux_modes[cz_idx * 2 + 1]]
            pq_instruction = cz_on_two_bosonic_qubits(mode_indices + aux_modes_this_cz)
            instructions.extend(pq_instruction)
            cz_idx += 1
        elif instruction.name == "p":
            mode = qubit_indices[0]
            instructions.extend(phase_gate_bosonic(instruction.params, mode))
        elif instruction.name == "measure":
            pq_instruction = pq.ParticleNumberMeasurement().on_modes(2 * qubit_indices[0], 2 * qubit_indices[0] + 1)
            instructions.append(pq_instruction)
        elif instruction.name == "if_else":
            raise ValueError("Conditional operations are not supported in dual-rail encoding.")
    return pq.Program(instructions=instructions)

def dual_rail_encode_from_qiskit(quantum_circuit: "QuantumCircuit") -> pq.Program:
    try:
        import qiskit
    except ImportError as e:
        raise ImportError("Qiskit package is not installed.") from e
    if not isinstance(quantum_circuit, qiskit.QuantumCircuit):
        raise TypeError(
            "The input argument to the dual_rail_encode_from_qiskit function should " \
            f"be a Qiskit QuantumCircuit, but it is of type '{type(quantum_circuit)}'."
        )
    return _encode_dual_rail_from_qiskit(quantum_circuit)

def get_bosonic_qubit_samples(raw_samples_for_modes: list[tuple]) -> list[tuple]:
    """Post-processes the raw samples from dual-rail encoded bosonic qubits.

    Args:
        raw_samples_for_modes: The raw samples obtained from the simulator.

    Returns:
        list[tuple]: The post-processed samples for the bosonic qubits.
    """
    if not any(len(samples) // 2 != 0 for samples in raw_samples_for_modes):
        raise ValueError(
            "The input raw_samples_for_modes should be a list of tuples, "
            f"where each tuple contains an outcome for each measured mode."
        )
    all_qubit_samples = []
    for samples_this_shot in raw_samples_for_modes:
        qubit_samples = []
        for i in range(0, len(samples_this_shot), 2):
            two_modes_outcome = [samples_this_shot[i], samples_this_shot[i + 1]]
            if two_modes_outcome == zero_bosonic_qubit_state:
                qubit_samples.append(0)
            elif two_modes_outcome == one_bosonic_qubit_state:
                qubit_samples.append(1)
            else:
                raise ValueError(f"Unexpected outcomes: {two_modes_outcome} for modes: {i}, {i+1}.")

        all_qubit_samples.append(tuple(qubit_samples))
    return all_qubit_samples