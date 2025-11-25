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

from piquasso.dual_rail_encoding import (
    dual_rail_encode_from_qiskit,
    get_bosonic_qubit_samples,
)
from qiskit import QuantumCircuit
import piquasso as pq
import numpy as np
import pytest


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
        assert create_photons.modes == (1,)

        # Check Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (0, 1)
        assert np.isclose(hadamard_1.params["theta"], np.pi / 4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (0,)

        measurement = prog.instructions[4]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (0, 1)

    def test_phase_gate(self):
        """Tests converting a circuit with a phase gate."""
        qc = QuantumCircuit(1, 1)
        qc.p(np.pi, 0)

        prog = dual_rail_encode_from_qiskit(qc)

        assert len(prog.instructions) == 3

        vacuum = prog.instructions[0]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (0, 1)

        create_photons = prog.instructions[1]
        assert isinstance(create_photons, pq.Create)
        assert create_photons.modes == (0,)

        phase_shift = prog.instructions[2]
        assert isinstance(phase_shift, pq.Phaseshifter)
        assert np.isclose(phase_shift.params["phi"], np.pi)
        assert phase_shift.modes == (1,)

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
        assert create_photons.modes == (1, 3, 4, 5)

        # Check 1. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (0, 1)
        assert np.isclose(hadamard_1.params["theta"], np.pi / 4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (0,)

        # Check 2. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[4]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (2, 3)
        assert np.isclose(hadamard_1.params["theta"], np.pi / 4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[5]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert hadamard_2.modes == (2,)
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
        first_bs_gate = prog.instructions[8]
        assert isinstance(first_bs_gate, pq.Beamsplitter)
        assert first_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[2])
        assert np.isclose(
            first_bs_gate.params["theta"], cz_beamsplitter_first_theta_value
        )

        sec_bs_gate = prog.instructions[9]
        assert isinstance(sec_bs_gate, pq.Beamsplitter)
        assert sec_bs_gate.modes == (bosonic_cz_indices[1], bosonic_cz_indices[3])
        assert np.isclose(
            sec_bs_gate.params["theta"], cz_beamsplitter_first_theta_value
        )

        third_bs_gate = prog.instructions[10]
        assert isinstance(third_bs_gate, pq.Beamsplitter)
        assert third_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[1])
        assert np.isclose(
            third_bs_gate.params["theta"], -1 * cz_beamsplitter_first_theta_value
        )

        fourth_bs_gate = prog.instructions[11]
        assert isinstance(fourth_bs_gate, pq.Beamsplitter)
        assert fourth_bs_gate.modes == (bosonic_cz_indices[2], bosonic_cz_indices[3])
        assert np.isclose(
            fourth_bs_gate.params["theta"], cz_beamsplitter_second_theta_value
        )

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

    def test_cnot_instructions(self):
        """Tests converting a circuit with a CNOT gate."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        prog = dual_rail_encode_from_qiskit(qc)

        assert len(prog.instructions) == 15

        vacuum = prog.instructions[0]
        assert isinstance(vacuum, pq.Vacuum)
        assert vacuum.modes == (0, 1, 2, 3, 4, 5)

        create_photons = prog.instructions[1]
        assert isinstance(create_photons, pq.Create)
        assert create_photons.modes == (1, 3, 4, 5)

        # Check 1. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[2]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (2, 3)
        assert np.isclose(hadamard_1.params["theta"], np.pi / 4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[3]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert np.isclose(hadamard_2.params["phi"], np.pi)
        assert hadamard_2.modes == (2,)

        # Check CZ dual-rail encoded implementations
        phase_shifter1 = prog.instructions[4]
        assert isinstance(phase_shifter1, pq.Phaseshifter)
        assert np.isclose(phase_shifter1.params["phi"], np.pi)

        phase_shifter2 = prog.instructions[5]
        assert isinstance(phase_shifter2, pq.Phaseshifter)
        assert np.isclose(phase_shifter2.params["phi"], np.pi)

        cz_beamsplitter_first_theta_value = 54.74 / 180 * np.pi
        cz_beamsplitter_second_theta_value = 17.63 / 180 * np.pi

        bosonic_cz_indices = [0, 2, 4, 5]
        first_bs_gate = prog.instructions[6]
        assert isinstance(first_bs_gate, pq.Beamsplitter)
        assert first_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[2])
        assert np.isclose(
            first_bs_gate.params["theta"], cz_beamsplitter_first_theta_value
        )

        sec_bs_gate = prog.instructions[7]
        assert isinstance(sec_bs_gate, pq.Beamsplitter)
        assert sec_bs_gate.modes == (bosonic_cz_indices[1], bosonic_cz_indices[3])
        assert np.isclose(
            sec_bs_gate.params["theta"], cz_beamsplitter_first_theta_value
        )

        third_bs_gate = prog.instructions[8]
        assert isinstance(third_bs_gate, pq.Beamsplitter)
        assert third_bs_gate.modes == (bosonic_cz_indices[0], bosonic_cz_indices[1])
        assert np.isclose(
            third_bs_gate.params["theta"], -1 * cz_beamsplitter_first_theta_value
        )

        fourth_bs_gate = prog.instructions[9]
        assert isinstance(fourth_bs_gate, pq.Beamsplitter)
        assert fourth_bs_gate.modes == (bosonic_cz_indices[2], bosonic_cz_indices[3])
        assert np.isclose(
            fourth_bs_gate.params["theta"], cz_beamsplitter_second_theta_value
        )

        post_selection = prog.instructions[10]
        assert isinstance(post_selection, pq.PostSelectPhotons)
        assert post_selection.params["photon_counts"] == [1, 1]
        assert post_selection.modes == (bosonic_cz_indices[2], bosonic_cz_indices[3])

        # Check 2. Hadamard dual-rail encoded implementations
        hadamard_1 = prog.instructions[11]
        assert isinstance(hadamard_1, pq.Beamsplitter)
        assert hadamard_1.modes == (2, 3)
        assert np.isclose(hadamard_1.params["theta"], np.pi / 4)
        assert np.isclose(hadamard_1.params["phi"], 0)

        hadamard_2 = prog.instructions[12]
        assert isinstance(hadamard_2, pq.Phaseshifter)
        assert hadamard_2.modes == (2,)
        assert np.isclose(hadamard_2.params["phi"], np.pi)

        measurement = prog.instructions[13]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (0, 1)

        measurement = prog.instructions[14]
        assert isinstance(measurement, pq.ParticleNumberMeasurement)
        assert measurement.modes == (2, 3)

    @pytest.mark.parametrize(
        "unsupported_gate_data",
        [("y", (1,)), ("swap", (0, 1))],
    )
    def test_invalid_gate_in_qiskit_circuit_raises(self, unsupported_gate_data):
        """Tests that an unsupported gate in QuantumCircuit raises a ValueError."""
        qc = QuantumCircuit(2)
        gate_name = unsupported_gate_data[0]
        getattr(qc, gate_name)(*unsupported_gate_data[1])
        with pytest.raises(
            ValueError,
            match=f"Unsupported instruction '{gate_name}' in the quantum circuit.",
        ):
            dual_rail_encode_from_qiskit(qc)


class TestIntegrationWithSimulator:
    """Tests the integration with the Simulator class."""

    @pytest.mark.parametrize("angle", np.linspace(0, np.pi, 4))
    def test_phase_gate(self, angle):
        """Tests that a Qiskit circuit with a phase gate executes as expected."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.p(angle, 0)

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=2, config=config, connector=connector)

        prog = pq.dual_rail_encoding.dual_rail_encode_from_qiskit(qc)
        res = simulator.execute(prog, shots=shots)

        assert np.isclose(res.state.fock_amplitudes_map[(1, 0)], 0)
        assert np.isclose(res.state.fock_amplitudes_map[(0, 1)], np.exp(1j*angle))

    def test_simulator_integration_cz_hadamard(self):
        """Tests that a Qiskit circuit with H and CZ can be executed."""
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

        assert len(res.branches) == 4

        outcomes = [
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 0, 1),
        ]
        for branch in res.branches:
            assert branch.outcome in outcomes
            assert np.isclose(float(branch.frequency), 0.25, atol=0.05)

    @pytest.mark.parametrize(
        "input_state, expected_state",
        [
            ((0, 1, 0, 1), (0, 1, 0, 1)),
            ((0, 1, 1, 0), (0, 1, 1, 0)),
            ((1, 0, 0, 1), (1, 0, 1, 0)),
            ((1, 0, 1, 0), (1, 0, 0, 1)),
        ],
    )
    def test_simulator_integration_cnot(self, input_state, expected_state):
        """Tests that a Qiskit circuit with H and CNOT can be executed."""
        qc = QuantumCircuit(2, 2)
        if input_state[0]:
            qc.x(0)
        if input_state[2]:
            qc.x(1)
        qc.cx(0, 1)

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=6, config=config, connector=connector)

        prog = dual_rail_encode_from_qiskit(qc)
        res = simulator.execute(prog, shots=shots)
        fock_amplitudes_map = res.state.fock_amplitudes_map
        for k, v in fock_amplitudes_map.items():
            if k != expected_state:
                assert np.isclose(fock_amplitudes_map[k], 0, atol=10e-4)
            else:
                assert not np.isclose(fock_amplitudes_map[k], 0)

    @pytest.mark.parametrize(
        "one_state, expected_outcome", [(False, (0, 1, 1, 0)), (True, (1, 0, 1, 0))]
    )
    def test_conditional_paulix(self, one_state, expected_outcome):
        """Tests that PauliX gate can be conditioned on measurement outcomes."""
        qc = QuantumCircuit(2, 2)
        if one_state:
            qc.x(0)
        qc.measure([0], [0])
        with qc.if_test((0, 0 if not one_state else 1)):
            qc.x(1)
        qc.measure([1], [1])

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=4, config=config, connector=connector)

        prog = dual_rail_encode_from_qiskit(qc)
        res = simulator.execute(prog, shots=shots)

        assert len(res.branches) == 1
        assert res.branches[0].outcome == expected_outcome
        assert np.isclose(float(res.branches[0].frequency), 1, atol=0.05)

    @pytest.mark.parametrize(
        "one_state, expected_outcome", [(False, (0, 1, 1, 0)), (True, (1, 0, 1, 0))]
    )
    def test_conditional_pauliz(self, one_state, expected_outcome):
        """Tests that PauliZ gate can be conditioned on measurement outcomes."""
        qc = QuantumCircuit(2, 2)
        if one_state:
            qc.x(0)
        qc.measure([0], [0])
        with qc.if_test((0, 0 if not one_state else 1)):
            qc.h(1)
            qc.z(1)
            qc.h(1)
        qc.measure([1], [1])

        connector = pq.NumpyConnector()
        cutoff = 8
        config = pq.Config(cutoff=cutoff)
        shots = 1000

        simulator = pq.PureFockSimulator(d=4, config=config, connector=connector)

        prog = dual_rail_encode_from_qiskit(qc)
        res = simulator.execute(prog, shots=shots)

        assert len(res.branches) == 1
        assert res.branches[0].outcome == expected_outcome
        assert np.isclose(float(res.branches[0].frequency), 1, atol=0.05)


raw_samples1 = [
    (0, 1),
    (1, 0),
]

raw_samples2 = [
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
]


class TestPostProcessing:
    """Tests the post-processing of measurement outcomes."""

    @pytest.mark.parametrize(
        "raw_samples, expected_qubit_samples",
        [
            (raw_samples1, [(0,), (1,)]),
            (raw_samples2, [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ],
    )
    def test_post_processing(self, raw_samples, expected_qubit_samples):
        """Tests the post-processing from dual-rail encoded bosonic qubits."""
        qubit_samples = get_bosonic_qubit_samples(raw_samples)
        assert qubit_samples == expected_qubit_samples
