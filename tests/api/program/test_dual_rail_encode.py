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

from piquasso.core.dual_rail_encoding import bosonic_hadamard, cz_on_two_bosonic_qubits, dual_rail_encode_from_qiskit, prep_bosonic_qubits
from qiskit import QuantumCircuit
import piquasso as pq
import numpy as np
import pytest

class TestDualRailEncodingInstructions:
    """Tests for the instructions used for dual rail encoding."""

    def test_prep_bosonic_qubit_instructions(self):
        """Tests the bosonic qubit preparation instruction."""
        all_modes = [0, 1]
        modes_with_one_photon = [0]
        instructions = prep_bosonic_qubits(all_modes, modes_with_one_photon)
        assert len(instructions) == 2

        vacuum = instructions[0]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (0, 1)

        create_photon = instructions[1]
        assert isinstance(create_photon, pq.Create)
        assert create_photon.modes == (0,)

    @pytest.mark.parametrize("all_modes, modes_with_one_photon, expected_to_have_amplitude",
                             [
                                 ([0, 1], [1], (0, 1)),
                                 ([0, 1, 2, 3], [1, 3], (0, 1, 0, 1)),
                             ])
    def test_prep_bosonic_qubits_amplitudes(self, all_modes, modes_with_one_photon, expected_to_have_amplitude):
        """Tests the amplitudes after bosonic qubit preparations."""
        instructions = prep_bosonic_qubits(all_modes, modes_with_one_photon)
        assert len(instructions) == 2

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=len(all_modes), config=config, connector=connector)

        prog = pq.Program(instructions=instructions)
        res = simulator.execute(prog, shots=shots)
        assert len(res.branches) == 1

        ampl_map = res.branches[0].state.fock_amplitudes_map
        assert len(ampl_map) == 1
        assert ampl_map[expected_to_have_amplitude] == 1

    @pytest.mark.parametrize("modes_with_one_photon, expected_amplitudes",
        [([1], (1 / np.sqrt(2), 1 / np.sqrt(2))),
        ([0], (1 / np.sqrt(2), -1 / np.sqrt(2))),
    ])
    def test_bosonic_hadamard_amplitudes(self, modes_with_one_photon, expected_amplitudes):
        """Tests the bosonic Hadamard gate implementation."""
        all_modes = [0, 1]
        instructions = prep_bosonic_qubits(all_modes, modes_with_one_photon)
        instructions.extend(bosonic_hadamard(*all_modes))
        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=len(all_modes), config=config, connector=connector)

        prog = pq.Program(instructions=instructions)
        res = simulator.execute(prog, shots=shots)
        amplitudes = res.branches[0].state.fock_amplitudes_map
        assert len(amplitudes) == 2
        zero_state = (0, 1)
        one_state = (1, 0)
        assert np.isclose(amplitudes[zero_state], expected_amplitudes[0])
        assert np.isclose(amplitudes[one_state], expected_amplitudes[1])

class TestDualRailEncoding:
    """Tests for the dual rail encoding functions."""

    def test_one_hadamard_one_measure(self):
        """Tests converting one circuit with Hadamard and measurement."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure([0], [0])

        prog = dual_rail_encode_from_qiskit(qc)

        assert len(prog.instructions) == 5
        vacuum = prog.instructions[0]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (0, 1)

        create_photons = prog.instructions[1]
        assert isinstance(create_photons, pq.Create)
        assert create_photons.modes == (0,)

        # Check Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (0, 1)
        assert np.isclose(hadamard_1.params["theta"], np.pi/4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (1,)

        measurement = prog.instructions[4]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (0, 1)

    def test_two_hadamards_and_cz(self):
        """Tests converting a circuit with two Hadamards and a CZ gate."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.cz(0, 1)
        qc.measure([0, 1], [0, 1])

        prog = dual_rail_encode_from_qiskit(qc)

        assert len(prog.instructions) == 15

        vacuum = prog.instructions[0]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (0, 1, 2, 3, 4, 5)

        create_photons = prog.instructions[1]
        assert isinstance(create_photons, pq.Create)
        assert create_photons.modes == (0, 2, 4, 5)

        # Check 1. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (0, 1)
        assert np.isclose(hadamard_1.params["theta"], np.pi/4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (1,)

        # Check 2. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[4]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (1, 3)
        assert np.isclose(hadamard_1.params["theta"], np.pi/4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[5]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert hadamard_2.modes == (3,)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        
        # Check CZ dual-rail encoded implementations
        phase_shifter1 = prog.instructions[6]
        assert isinstance(phase_shifter1, pq.Phaseshifter)
        assert np.isclose(phase_shifter1.params["phi"], np.pi)

        phase_shifter2 = prog.instructions[7]
        assert isinstance(phase_shifter2, pq.Phaseshifter)
        assert np.isclose(phase_shifter2.params["phi"], np.pi)

        cz_beamsplitter_first_theta_value = 54.74 / 180 * np.pi
        cz_beamsplitter_second_theta_value = 17.63 / 180 * np.pi

        bosonic_cz_indices = [0, 2, 4, 5]
        first_bs_gate =  prog.instructions[8]
        assert isinstance(first_bs_gate, pq.Beamsplitter)
        assert first_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[2])
        assert np.isclose(first_bs_gate.params["theta"], cz_beamsplitter_first_theta_value)

        sec_bs_gate =  prog.instructions[9]
        assert isinstance(sec_bs_gate, pq.Beamsplitter)
        assert sec_bs_gate.modes == (bosonic_cz_indices[1], bosonic_cz_indices[3])
        assert np.isclose(sec_bs_gate.params["theta"], cz_beamsplitter_first_theta_value)

        third_bs_gate =  prog.instructions[10]
        assert isinstance(third_bs_gate, pq.Beamsplitter)
        assert third_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[1])
        assert np.isclose(third_bs_gate.params["theta"], -1*cz_beamsplitter_first_theta_value)

        fourth_bs_gate =  prog.instructions[11]
        assert isinstance(fourth_bs_gate, pq.Beamsplitter)
        assert fourth_bs_gate.modes == (bosonic_cz_indices[2], bosonic_cz_indices[3])
        assert np.isclose(fourth_bs_gate.params["theta"], cz_beamsplitter_second_theta_value)
        
        post_selection = prog.instructions[12]
        assert isinstance(post_selection, pq.PostSelectPhotons)
        assert post_selection.params["photon_counts"] == [1, 1]
        assert post_selection.modes == (bosonic_cz_indices[2], bosonic_cz_indices[3])

        measurement = prog.instructions[13]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (0, 1)

        measurement = prog.instructions[14]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (2, 3)

    @pytest.mark.parametrize("unsupported_gate_data",
                             [("x",(0,)), ("y", (1,)), ("z", (0,)), ("cx", (0,1)), ("swap", (0,1))]
                             )
    def test_invalid_gate_in_qiskit_circuit_raises(self, unsupported_gate_data):
        """Tests that an unsupported gate in the Qiskit QuantumCircuit raises a ValueError."""
        qc = QuantumCircuit(2)
        gate_name = unsupported_gate_data[0]
        getattr(qc, gate_name)(*unsupported_gate_data[1])
        with pytest.raises(ValueError, match=f"Unsupported instruction '{gate_name}' in the quantum circuit."):
            dual_rail_encode_from_qiskit(qc)


class TestIntegrationWithSimulator:
    """Tests the integration with the Simulator class."""

    def test_simulator_integration(self):
        """Tests that a Qiskit circuit can be executed."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.cz(0, 1)
        qc.measure([0, 1], [0, 1])

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=6, config=config, connector=connector)

        prog = dual_rail_encode_from_qiskit(qc)
        res = simulator.execute(prog, shots=shots)