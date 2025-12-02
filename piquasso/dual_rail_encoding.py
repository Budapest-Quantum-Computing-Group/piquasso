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


r"""Dual-rail encoding utilities for bosonic qubits.

This module provides functions to encode qubit circuits defined in Qiskit into dual-rail
bosonic qubit Piquasso programs.

For the dual-rail encoding, the following convention is used:

.. math::
    |0\rangle_{\text{qubit}} = \ket{1, 0}_{\text{qumodes}} \\
    |1\rangle_{\text{qubit}} = \ket{0, 1}_{\text{qumodes}}
"""

import piquasso as pq
import numpy as np

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

_zero_bosonic_qubit_state = [1, 0]
_one_bosonic_qubit_state = [0, 1]


def _prep_bosonic_qubits(
    all_modes: List[int], modes_with_one_photon: List[int]
) -> List[pq.Instruction]:
    r"""Prepares a bosonic qubits in the specified basis states.

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


def _paulix_bosonic(mode1, mode2):
    """Applies a Pauli-X gate on a bosonic qubit encoded in dual-rail format.

    Args:
        mode: The mode of the bosonic qubit.
    """
    instructions = []
    instructions.append(pq.Phaseshifter(np.pi).on_modes(mode2))
    instructions.append(pq.Beamsplitter(np.pi / 2).on_modes(mode1, mode2))
    return instructions


def _pauliy_bosonic(mode1, mode2):
    """Applies a Pauli-Y gate on a bosonic qubit encoded in dual-rail format.

    Args:
        mode: The mode of the bosonic qubit.
    """
    instructions = []
    instructions.append(pq.Beamsplitter(-np.pi / 2, np.pi / 2).on_modes(mode1, mode2))
    instructions.append(pq.Phaseshifter(np.pi).on_modes(mode2))
    return instructions

def _pauliz_bosonic(mode1, mode2):
    """Applies a Pauli-Z gate on a bosonic qubit encoded in dual-rail format.

    Args:
        mode: The mode of the bosonic qubit.
    """
    return _phase_gate_bosonic(np.pi, mode2)

def _rx_bosonic(theta, mode1, mode2):
    """Applies a PauliX-rotation gate on a bosonic qubit encoded in dual-rail format.

    Args:
        theta: The phase angle.
        mode: The first mode of the bosonic qubit (the second mode is mode + 1).

    Returns:
        A list of Piquasso instructions implementing the phase gate.
    """
    instructions = []
    instructions.append(pq.Beamsplitter(theta / 2, -np.pi / 2).on_modes(mode1, mode2))
    return instructions


def _phase_gate_bosonic(theta, mode):
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


def _hadamard_bosonic(mode1, mode2):
    instructions = []
    instructions.append(pq.Phaseshifter(np.pi).on_modes(mode2))
    instructions.append(pq.Beamsplitter(np.pi / 4).on_modes(mode1, mode2))
    return instructions


def _cz_on_two_bosonic_qubits(modes):
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
        pq.Beamsplitter(-1 * cz_beamsplitter_first_theta_value).on_modes(
            modes[0], modes[1]
        )
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


def _cnot_on_two_bosonic_qubits(modes):
    """Note: requires two auxiliary modes with one photon each."""
    instructions = []

    # Apply Hadamard on the control bosonic qubit
    H_instruction_1 = _hadamard_bosonic(modes[2], modes[3])

    # Extract the two data modes and two aux modes
    cz_modes = [modes[1], modes[3], modes[4], modes[5]]
    CZ_instruction = _cz_on_two_bosonic_qubits(cz_modes)

    # Apply Hadamard on the control bosonic qubit
    H_instruction_2 = _hadamard_bosonic(modes[2], modes[3])

    instructions.extend(H_instruction_1)
    instructions.extend(CZ_instruction)
    instructions.extend(H_instruction_2)
    return instructions


def _get_condition_function(qubit_index, measurement_value):
    """Returns a condition for conditional operations based on measurement outcomes.

    This function converts the qubit index to the corresponding dual-rail mode
    indices and checks the measurement outcomes of those modes to determine the
    qubit measurement outcome.
    """

    def condition(outcomes):
        two_mode_outcomes = [(outcomes[qubit_index * 2], outcomes[qubit_index * 2 + 1])]
        qubit_outcome = get_bosonic_qubit_samples(two_mode_outcomes)[0][0]
        return qubit_outcome == measurement_value

    return condition


def _map_qiskit_instr_to_pq(qiskit_instruction, modes, aux_modes):
    instruction_name = qiskit_instruction.name
    instructions = []
    if instruction_name == "h":
        pq_instruction = _hadamard_bosonic(modes[0], modes[1])
        instructions.extend(pq_instruction)
    elif instruction_name == "x":
        pq_instruction = _paulix_bosonic(modes[0], modes[1])
        instructions.extend(pq_instruction)
    elif instruction_name == "y":
        pq_instruction = _pauliy_bosonic(modes[0], modes[1])
        instructions.extend(pq_instruction)
    elif instruction_name == "z":
        pq_instruction = _pauliz_bosonic(modes[0], modes[1])
        instructions.extend(pq_instruction)
    elif instruction_name == "cz":
        pq_instruction = _cz_on_two_bosonic_qubits(modes + aux_modes)
        instructions.extend(pq_instruction)
    elif instruction_name == "cx":
        pq_instruction = _cnot_on_two_bosonic_qubits(modes + aux_modes)
        instructions.extend(pq_instruction)
    elif instruction_name == "rx":
        instructions.extend(_rx_bosonic(qiskit_instruction.params[0], modes[0], modes[1]))
    elif instruction_name == "p":
        instructions.extend(_phase_gate_bosonic(qiskit_instruction.params[0], modes[1]))
    elif instruction_name == "measure":
        pq_instruction = pq.ParticleNumberMeasurement().on_modes(modes[0], modes[1])
        instructions.append(pq_instruction)
    elif instruction_name == "if_else":
        true_branch_instructions = qiskit_instruction.operation.params[0]

        cond = qiskit_instruction.operation.condition

        condition = _get_condition_function(cond[0]._index, cond[1])
        for inner_instr_qiskit in true_branch_instructions:
            instr_list = _map_qiskit_instr_to_pq(inner_instr_qiskit, modes, aux_modes)
            for instr in instr_list:
                instructions.append(instr.when(condition))
    else:
        raise ValueError(
            f"Unsupported instruction '{instruction_name}' in the quantum circuit."
        )
    return instructions


def _encode_dual_rail_from_qiskit(qc):
    """The function to encode a QuantumCircuit into dual-rail instructions."""
    num_cz = sum(1 for instruction in qc.data if instruction.name in ("cz", "cx"))
    num_bosonic_qubits = qc.num_qubits

    idx_one_photon = np.where(np.array(_zero_bosonic_qubit_state) == 1)[0][0]
    modes_with_one_photon = list(
        range(idx_one_photon, num_bosonic_qubits * 2 + idx_one_photon, 2)
    )
    num_aux_needed = num_cz * 2
    aux_modes_all = list(
        range(num_bosonic_qubits * 2, num_bosonic_qubits * 2 + num_aux_needed)
    )

    all_modes = list(range(num_bosonic_qubits * 2)) + aux_modes_all

    instructions = []

    modes_with_one_photon = modes_with_one_photon + aux_modes_all
    preparations = _prep_bosonic_qubits(all_modes, modes_with_one_photon)
    instructions.extend(preparations)

    cz_idx = 0
    for instr_qiskit in qc.data:
        qubit_indices = [qc.find_bit(q).index for q in instr_qiskit.qubits]

        if instr_qiskit.name in ("cz", "cx"):
            if instr_qiskit.name == "cz":
                modes = [2 * qubit_indices[0] + 1, 2 * qubit_indices[1] + 1]
            else:
                modes = [
                    2 * qubit_indices[0],
                    2 * qubit_indices[0] + 1,
                    2 * qubit_indices[1],
                    2 * qubit_indices[1] + 1,
                ]
            aux_modes = [aux_modes_all[cz_idx * 2], aux_modes_all[cz_idx * 2 + 1]]
            cz_idx += 1
        else:
            qubit = qubit_indices[0]
            modes = [2 * qubit, 2 * qubit + 1]
            aux_modes = []
        mapped_instructions = _map_qiskit_instr_to_pq(instr_qiskit, modes, aux_modes)
        instructions.extend(mapped_instructions)

    return instructions


def dual_rail_encode_from_qiskit(quantum_circuit: "QuantumCircuit") -> pq.Program:
    """Encodes a Qiskit QuantumCircuit into a dual-rail bosonic qubit Piquasso program.

    Args:
        quantum_circuit: The Qiskit QuantumCircuit to be encoded.

    Returns:
        A Piquasso Program representing the dual-rail encoded bosonic qubit circuit.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise ImportError("Qiskit package is not installed.") from e
    if not isinstance(quantum_circuit, QuantumCircuit):
        raise TypeError(
            "The input argument to the dual_rail_encode_from_qiskit function should "
            f"be a Qiskit QuantumCircuit, but it is of type '{type(quantum_circuit)}'."
        )
    instructions = _encode_dual_rail_from_qiskit(quantum_circuit)
    return pq.Program(instructions=instructions)


def get_bosonic_qubit_samples(raw_samples_for_modes: List[tuple]) -> List[tuple]:
    """Post-processes the raw samples from dual-rail encoded bosonic qubits.

    Args:
        raw_samples_for_modes: The raw samples obtained from the simulator.

    Returns:
        list[tuple]: The post-processed samples for the bosonic qubits.
    """
    if not any(len(samples) // 2 != 0 for samples in raw_samples_for_modes):
        raise ValueError(
            "The input raw_samples_for_modes should be a list of tuples, "
            "where each tuple contains an outcome for each measured mode."
        )
    all_qubit_samples = []
    for samples_this_shot in raw_samples_for_modes:
        qubit_samples = []
        for i in range(0, len(samples_this_shot), 2):
            two_modes_outcome = [samples_this_shot[i], samples_this_shot[i + 1]]
            if two_modes_outcome == _zero_bosonic_qubit_state:
                qubit_samples.append(0)
            elif two_modes_outcome == _one_bosonic_qubit_state:
                qubit_samples.append(1)
            else:
                raise ValueError(
                    f"Unexpected outcomes: {two_modes_outcome} for modes: {i}, {i+1}."
                )

        all_qubit_samples.append(tuple(qubit_samples))
    return all_qubit_samples
