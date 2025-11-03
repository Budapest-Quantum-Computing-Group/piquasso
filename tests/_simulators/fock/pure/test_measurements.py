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

import numpy as np
import pytest

import piquasso as pq


def test_measure_particle_number_on_one_mode():
    cutoff = 3
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1, 1]) * np.sqrt(2 / 6)

        pq.Q(2) | pq.StateVector([1]) * np.sqrt(1 / 6)
        pq.Q(2) | pq.StateVector([2]) * np.sqrt(3 / 6)

        pq.Q(2) | pq.ParticleNumberMeasurement()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=cutoff))

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (1,) or sample == (2,)

    if sample == (1,):
        expected_simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff - 1)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[
                0.5773502691896258 * pq.StateVector([0, 0]),
                0.816496580927726 * pq.StateVector([0, 1]),
            ]
        ).state

    elif sample == (2,):
        expected_simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0, 0])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_two_modes():
    cutoff = 3
    with pq.Program() as program:
        pq.Q(1, 2) | pq.StateVector([1, 1]) * np.sqrt(2 / 6)
        pq.Q(1, 2) | pq.StateVector([0, 1]) * np.sqrt(1 / 6)
        pq.Q(1, 2) | pq.StateVector([0, 2]) * np.sqrt(3 / 6)

        pq.Q(1, 2) | pq.ParticleNumberMeasurement()

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=cutoff))

    result = simulator.execute(program)

    assert np.isclose(sum(result.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 1) or sample == (1, 1) or sample == (0, 2)

    if sample == (0, 1):
        expected_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff - 1)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0])]
        ).state

    elif sample == (1, 1):
        expected_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0])]
        ).state

    elif sample == (0, 2):
        expected_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.StateVector([0])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_all_modes():
    config = pq.Config(cutoff=2)

    simulator = pq.PureFockSimulator(d=3, config=config)

    with pq.Program() as program:
        pq.Q() | 0.5 * pq.StateVector([0, 0, 0])
        pq.Q() | 0.5 * pq.StateVector([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.StateVector([1, 0, 0])

        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program)

    sample = result.samples[0]
    assert sample == (0, 0, 0) or sample == (1, 0, 0) or sample == (0, 0, 1)

    assert result.state is None


def test_measure_particle_number_with_multiple_shots():
    shots = 4

    # TODO: This is very unusual, that we need to know the cutoff for specifying the
    # state. It should be imposed, that the only parameter for a state should be `d` and
    #  `config` maybe.
    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=2))

    with pq.Program() as program:
        pq.Q() | 0.5 * pq.StateVector([0, 0, 0])
        pq.Q() | 0.5 * pq.StateVector([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.StateVector([1, 0, 0])

        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots


def test_HomodyneMeasurement_one_mode():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=20, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=1.0)

        pq.Q(0) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            (0.9948477359537836,),
            (0.8961788544207923,),
            (0.7786192340830824,),
            (0.7558150016761611,),
            (1.8723492374692325,),
            (2.073005981623021,),
            (0.9259892199775892,),
            (2.2811081227922023,),
            (2.040472494494099,),
            (1.4372232395859554,),
            (1.6487776718216636,),
            (2.4239266293258774,),
            (1.7496043479608292,),
            (2.4442999983811964,),
            (1.446437822064922,),
            (2.007139616291834,),
            (0.853145932711283,),
            (0.8690238503020433,),
            (2.0608290677018624,),
            (0.2766097227293141,),
        ],
    )


def test_HomodyneMeasurement_two_modes():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=7, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)

        pq.Q(0, 1) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            (1.1652567241413503, -0.4599721082577378),
            (1.1420679631869184, -1.1222243286933127),
            (1.7169027959544987, -1.1687018442467523),
            (1.353781344741021, 0.15965134967702863),
            (1.6941244813802323, -0.5271629121478212),
            (0.07141270351567434, -2.248135143359257),
            (0.7031530068841597, -0.5197549770778916),
            (1.3000800741077214, -0.6858102208990626),
            (0.730065617791506, -1.1609622490424483),
            (0.18825745705786992, -1.3957717940806902),
            (1.6872341928509342, 0.026887748866724386),
            (1.3659607608300393, -1.294942798258335),
            (1.0424824437006375, -1.8439813778948924),
            (0.1566102396117092, 0.02666196714850383),
            (0.3351489720026158, -0.6550974166745),
            (1.2948514266475415, 0.08064954724023134),
            (0.6558635109854968, -0.24068383925038866),
            (0.04870163370638721, -0.14112144328518797),
            (1.7372732597359886, -1.2657744166963962),
            (0.16191613301463356, -1.3493562838419095),
        ],
    )


def test_HomodyneMeasurement_two_modes_with_1_mode_sampled():
    shots = 20

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=7, seed_sequence=123, hbar=1)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)

        pq.Q(0) | pq.HomodyneMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots

    assert np.allclose(
        result.samples,
        [
            (0.2877500173927323,),
            (0.1890751973948675,),
            (0.07150248898484375,),
            (0.04870163370638658,),
            (1.1652567241413503,),
            (1.3659607608300393,),
            (0.21888906261992586,),
            (1.5740900465364953,),
            (1.3334204312850824,),
            (0.7300656177915056,),
            (0.9416360445806452,),
            (1.7169027959544987,),
            (1.0424824437006375,),
            (1.7372732597359888,),
            (0.7392800094998989,),
            (1.3000800741077216,),
            (0.14603568513374862,),
            (0.16191613301463317,),
            (1.353781344741021,),
            (-0.430574768916639,),
        ],
    )


def test_HomodyneMeasurement_different_hbar_values():
    shots = 20

    simulator_hbar_2 = pq.PureFockSimulator(
        d=3, config=pq.Config(cutoff=7, seed_sequence=123, hbar=2)
    )
    simulator_hbar_3 = pq.PureFockSimulator(
        d=3, config=pq.Config(cutoff=7, seed_sequence=123, hbar=3)
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5)
        pq.Q(1) | pq.Displacement(r=-0.5)
        pq.Q(2) | pq.Squeezing(r=0.1)
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0, 2) | pq.HomodyneMeasurement()

    samples_hbar_2 = simulator_hbar_2.execute(program, shots).samples
    samples_hbar_3 = simulator_hbar_3.execute(program, shots).samples

    assert np.allclose(samples_hbar_2 / np.sqrt(2), samples_hbar_3 / np.sqrt(3))


def test_ParticleNumberMeasurement_resulting_state():
    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1])

        pq.Q() | pq.Beamsplitter5050()

        pq.Q(0) | pq.ParticleNumberMeasurement()

    result = simulator.execute(program)

    assert result.state.d == 1
    assert np.isclose(sum(result.state.fock_probabilities), 1)


class TestMidCircuitMeasurements:
    """Test programs that contain mid-circuit measurements."""

    def test_multi_ParticleNumberMeasurement_in_one_program(self):
        simulator = pq.PureFockSimulator(
            d=4, config=pq.Config(cutoff=3, seed_sequence=123)
        )

        with pq.Program() as program:
            pq.Q() | pq.StateVector([0, 1, 1, 0])

            pq.Q(0, 1) | pq.Beamsplitter5050()
            pq.Q(0) | pq.ParticleNumberMeasurement()

            pq.Q(1, 2) | pq.Beamsplitter5050()
            pq.Q(1) | pq.ParticleNumberMeasurement()

            pq.Q(2, 3) | pq.Beamsplitter5050()

            pq.Q(2) | pq.ParticleNumberMeasurement()
            pq.Q(3) | pq.ParticleNumberMeasurement()

        result = simulator.execute(program, shots=10)

        assert np.allclose(
            result.samples,
            [
                (1, 0, 1, 0),
                (1, 0, 0, 1),
                (0, 0, 0, 2),
                (1, 0, 1, 0),
                (0, 0, 2, 0),
                (0, 0, 0, 2),
                (0, 2, 0, 0),
                (0, 0, 2, 0),
                (0, 0, 0, 2),
                (0, 0, 2, 0),
            ],
        )

    @pytest.mark.parametrize("input_modes", [[], [0, 1, 2]])
    @pytest.mark.parametrize(
        "res_samples", [(1, 0, 1), (0, 0, 1), (1, 1, 0), (2, 0, 1)]
    )
    def test_post_select_and_pnm(self, input_modes, res_samples):

        with pq.Program() as program:
            pq.Q() | pq.StateVector(
                [res_samples[0], res_samples[1], res_samples[2], 0, 0]
            )
            pq.Q(3, 4) | pq.PostSelectPhotons(photon_counts=[0, 0])
            pq.Q(0) | pq.Squeezing(0.0)
            pq.Q(*input_modes) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(d=5)
        res = simulator.execute(program, shots=1)
        assert res.samples == [res_samples]

    def test_imperfect_post_select_and_pnm(self):
        d = 7
        cutoff = 7

        detector_efficiency_matrix = np.array(
            [
                [1.0, 0.2, 0.1],
                [0.0, 0.8, 0.2],
                [0.0, 0.0, 0.7],
            ]
        )

        coeffs = np.sqrt([0.1, 0.3, 0.4, 0.05, 0.1, 0.05])

        with pq.Program() as program:
            pq.Q() | pq.StateVector([0, 0, 0, 2, 1, 1, 2]) * coeffs[0]
            pq.Q() | pq.StateVector([0, 0, 2, 0, 1, 1, 2]) * coeffs[1]
            pq.Q() | pq.StateVector([0, 1, 0, 1, 1, 1, 2]) * coeffs[2]
            pq.Q() | pq.StateVector([1, 1, 0, 1, 0, 1, 2]) * coeffs[3]
            pq.Q() | pq.StateVector([3, 0, 0, 0, 0, 1, 2]) * coeffs[4]

            pq.Q(5, 6) | pq.ParticleNumberMeasurement()
            pq.Q(2) | pq.Squeezing(0.0)
            pq.Q(2, 4) | pq.ImperfectPostSelectPhotons(
                photon_counts=(0, 1),
                detector_efficiency_matrix=detector_efficiency_matrix,
            )

        simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

        samples = simulator.execute(program).samples
        assert samples == [(1, 2)]

    @pytest.mark.parametrize("res_samples", [(1, 0), (0, 1), (3, 0), (0, 3)])
    def test_reindexing_for_measurements_explicit_modes(self, res_samples):
        """Test a case where internally mode reindexing happens for active modes."""
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, res_samples[0], 0, res_samples[1]])
            pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 1])
            pq.Q(2) | pq.Squeezing(0.0)
            pq.Q(2, 4) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(d=5, config=pq.Config(cutoff=6))
        res = simulator.execute(program, shots=1)
        assert res.samples == [res_samples]

    @pytest.mark.parametrize("measured_mode", [0, 1])
    def test_measuring_inactive_raises(self, measured_mode):
        """Test that measuring inactive modes raises an error."""
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
            pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 1])
            pq.Q(2) | pq.Squeezing(0.0)
            pq.Q(measured_mode) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(d=5)
        with pytest.raises(ValueError, match=f"are not active: {{{measured_mode}}}"):
            simulator.execute(program, shots=1)

    def test_mid_circuit_not_allowed(self):
        """
        Test that an error is raised for mid-circuit measurements that are not allowed.
        """
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
            pq.Q(0, 1) | pq.HomodyneMeasurement()
            pq.Q(2) | pq.Squeezing(0.0)
            pq.Q(2) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(d=5)
        with pytest.raises(
            pq.api.exceptions.InvalidSimulation,
            match="not allowed as a mid-circuit measurement",
        ):
            simulator.execute(program, shots=1)


def test_conditional_squeezing():
    r = 0.2

    program = pq.Program(
        instructions=[
            pq.StateVector([0, 2]) * np.sqrt(1 / 2),
            pq.StateVector([2, 0]) * np.sqrt(1 / 2),
            pq.ParticleNumberMeasurement().on_modes(1),
            pq.Squeezing(r=r).on_modes(0).when(lambda x: x[-1] == 2),
        ]
    )

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=7, seed_sequence=123))

    result = simulator.execute(program, shots=10)

    expected_squeezed_state = result.branches[1].state

    actual_squeezed_state = (
        pq.PureFockSimulator(d=1, config=pq.Config(cutoff=5))
        .execute_instructions([pq.Vacuum(), pq.Squeezing(r=r)])
        .state
    )

    assert expected_squeezed_state == actual_squeezed_state
