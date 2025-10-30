from piquasso.core._dual_rail_encoding import DualRailConverter
from qiskit import QuantumCircuit
import piquasso as pq
import numpy as np
import pytest

class TestDualRailConverter:
    """Tests for the DualRailConverter class."""

    def test_convert_from_qiskit_qc(self):
        """Tests the conversion from a Qiskit QuantumCircuit to a Piquasso Program with dual-rail encoding."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.cz(0, 1)
        qc.measure([0, 1], [0, 1])

        converter = DualRailConverter()
        prog = converter._convert_from_qiskit_qc(qc)

        assert len(prog.instructions) == 15

        create_photons = prog.instructions[0]
        assert isinstance(create_photons, pq.Create)
        assert create_photons.modes == (0, 2, 4, 5)

        vacuum = prog.instructions[1]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (1, 3)

        # Check 1. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (0, 2)
        assert np.isclose(hadamard_1.params["theta"], np.pi/4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (2,)

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
        converter = DualRailConverter()
        with pytest.raises(ValueError, match=f"Unsupported instruction '{gate_name}' in the quantum circuit."):
            converter._convert_from_qiskit_qc(qc)