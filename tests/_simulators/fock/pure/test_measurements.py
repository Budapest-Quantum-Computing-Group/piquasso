#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
        pq.Q() | pq.NumberState([0, 1, 1]) * np.sqrt(2 / 6)

        pq.Q(2) | pq.NumberState([1]) * np.sqrt(1 / 6)
        pq.Q(2) | pq.NumberState([2]) * np.sqrt(3 / 6)

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
                0.5773502691896258 * pq.NumberState([0, 0]),
                0.816496580927726 * pq.NumberState([0, 1]),
            ]
        ).state

    elif sample == (2,):
        expected_simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.NumberState([0, 0])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_two_modes():
    cutoff = 3
    with pq.Program() as program:
        pq.Q(1, 2) | pq.NumberState([1, 1]) * np.sqrt(2 / 6)
        pq.Q(1, 2) | pq.NumberState([0, 1]) * np.sqrt(1 / 6)
        pq.Q(1, 2) | pq.NumberState([0, 2]) * np.sqrt(3 / 6)

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
            instructions=[pq.NumberState([0])]
        ).state

    elif sample == (1, 1):
        expected_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.NumberState([0])]
        ).state

    elif sample == (0, 2):
        expected_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff - 2)
        )
        expected_state = expected_simulator.execute_instructions(
            instructions=[pq.NumberState([0])]
        ).state

    assert result.state == expected_state


def test_measure_particle_number_on_all_modes():
    config = pq.Config(cutoff=2)

    simulator = pq.PureFockSimulator(d=3, config=config)

    with pq.Program() as program:
        pq.Q() | 0.5 * pq.NumberState([0, 0, 0])
        pq.Q() | 0.5 * pq.NumberState([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.NumberState([1, 0, 0])

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
        pq.Q() | 0.5 * pq.NumberState([0, 0, 0])
        pq.Q() | 0.5 * pq.NumberState([0, 0, 1])
        pq.Q() | np.sqrt(1 / 2) * pq.NumberState([1, 0, 0])

        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots)

    assert len(result.samples) == shots


class TestHomodyneMeasurement:
    """Test programs that contain homodyne measurements."""

    def test_one_mode(self):
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

    def test_two_modes(self):
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
                (1.1652567241413498, -0.4724674262034825),
                (1.142067963186918, -1.1242684346801375),
                (1.7169027959544978, -1.1272989266942388),
                (1.3537813447410212, 0.1600572913698543),
                (1.694124481380232, -0.4815269999628965),
                (0.0714127035156745, -2.2437165482844605),
                (0.7031530068841597, -0.5588987236232392),
                (1.3000800741077216, -0.6752213864792986),
                (0.7300656177915056, -1.195230359739771),
                (0.1882574570578703, -1.3929139375890263),
                (1.6872341928509336, 0.07166752771911794),
                (1.365960760830039, -1.2690693618266498),
                (1.0424824437006373, -1.8456532368070406),
                (0.1566102396117087, 0.07594455581931851),
                (0.33514897200261545, -0.6587150007635347),
                (1.2948514266475415, 0.07388109980199688),
                (0.6558635109854966, -0.27709353836190037),
                (0.048701633706386645, -0.0814292603551763),
                (1.7372732597359892, -1.2258538004774424),
                (0.1619161330146332, -1.3422261054741327),
            ],
        )

    def test_two_modes_with_custom_angles(self):
        shots = 20

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=7, seed_sequence=123, hbar=1)
        )

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=0.5)
            pq.Q(1) | pq.Displacement(r=-0.5)

            pq.Q(0, 1) | pq.HomodyneMeasurement(phi=np.array([np.pi / 3, np.pi / 7]))

        result = simulator.execute(program, shots)

        assert len(result.samples) == shots

        assert np.allclose(
            result.samples,
            [
                (0.8115895087144187, -0.4023365317412277),
                (0.7884051896171809, -1.053345302809607),
                (1.3633657516508488, -1.0562677344995),
                (1.000103000542421, 0.23019260763417856),
                (1.3405716317499146, -0.41092018285986515),
                (-0.28210409107251755, -2.1732377487042682),
                (0.34959166449178997, -0.48935907929919825),
                (0.9463998112984134, -0.6046904039097345),
                (0.3765025629265892, -1.1250293045767068),
                (-0.16526150121841027, -1.3235175114013649),
                (1.333676748860502, 0.14191238583582103),
                (1.0122834770081088, -1.1977533597526133),
                (0.6888440037179449, -1.7740791146844932),
                (-0.1969062603847216, 0.1462934923376366),
                (-0.018405377145828, -0.5893729293894905),
                (0.9411712100209174, 0.14395035684043403),
                (0.30230200482446273, -0.2075567163036498),
                (-0.30482049730236577, -0.011094471676292508),
                (1.3837511528845214, -1.154924531043401),
                (-0.19160074598689744, -1.2729029203181923),
            ],
        )

    def test_two_modes_with_1_mode_sampled(self):
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

    def test_different_hbar_values(self):
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

            pq.Q(0, 2) | pq.HomodyneMeasurement(phi=np.array([np.pi / 3, np.pi / 7]))

        samples_hbar_2 = simulator_hbar_2.execute(program, shots).samples
        samples_hbar_3 = simulator_hbar_3.execute(program, shots).samples

        assert np.allclose(samples_hbar_2 / np.sqrt(2), samples_hbar_3 / np.sqrt(3))

    def test_post_measurement_state_of_product_state(self):
        """Measuring an uncorrelated vacuum mode must not change the other mode."""
        shots = 10
        cutoff = 7

        displacement_r = 0.3
        displacement_phi = 0.2
        homodyne_phi = 0.7

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff, seed_sequence=123, hbar=1)
        )

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(1) | pq.Displacement(r=displacement_r, phi=displacement_phi)

            pq.Q(0) | pq.HomodyneMeasurement(phi=homodyne_phi)

        result = simulator.execute(program, shots=shots)

        reference_simulator = pq.PureFockSimulator(
            d=1, config=pq.Config(cutoff=cutoff, hbar=1)
        )

        with pq.Program() as reference_program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=displacement_r, phi=displacement_phi)

        reference_state = reference_simulator.execute(reference_program).state

        assert len(result.branches) == shots

        for branch in result.branches:
            post_measurement_state = branch.state

            assert post_measurement_state is not None
            assert post_measurement_state.d == 1

            assert np.isclose(reference_state.fidelity(post_measurement_state), 1.0)

    def test_post_measurement_state_of_entangled_state(self):
        """Check the conditional state against the analytical projection.

        Prepare

            (|0, 0> + |1, 1>) / sqrt(2)

        and homodyne-measure mode 0.

        For dimensionless outcome q,

            <q; phi | psi>
                proportional to
            |0> + sqrt(2) q exp(-i phi) |1>.
        """
        shots = 10
        cutoff = 3
        hbar = 1.0
        phi = np.pi / 5

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff, seed_sequence=123, hbar=hbar)
        )

        with pq.Program() as program:
            pq.Q() | (pq.NumberState([0, 0]) + pq.NumberState([1, 1])) / np.sqrt(2.0)

            pq.Q(0) | pq.HomodyneMeasurement(phi=phi)

        result = simulator.execute(
            program,
            shots=shots,
        )

        assert len(result.branches) == shots

        for branch in result.branches:
            post_measurement_state = branch.state

            assert post_measurement_state is not None
            assert post_measurement_state.d == 1

            outcome = branch.outcome[0]
            q = outcome / np.sqrt(hbar)

            expected = np.zeros(cutoff, dtype=complex)

            expected[0] = 1.0
            expected[1] = np.sqrt(2.0) * q * np.exp(-1j * phi)

            expected /= np.linalg.norm(expected)

            overlap = np.vdot(
                expected,
                post_measurement_state.state_vector,
            )

            fidelity = np.abs(overlap) ** 2

            assert np.isclose(fidelity, 1.0)

    def test_post_measurement_state_when_measuring_middle_mode(self):
        """Check that removing a non-edge mode preserves mode ordering."""
        shots = 10
        cutoff = 4

        simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=cutoff, seed_sequence=123, hbar=1)
        )

        # |1, 0, 2>
        #
        # Mode 1 is vacuum, so homodyne measurement of mode 1
        # must leave |1, 2> on the remaining modes.
        with pq.Program() as program:
            pq.Q() | pq.NumberState([1, 0, 2])

            pq.Q(1) | pq.HomodyneMeasurement(phi=np.pi / 3)

        result = simulator.execute(program, shots=shots)

        reference_simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=cutoff, hbar=1)
        )

        with pq.Program() as reference_program:
            pq.Q() | pq.NumberState([1, 2])

        reference_state = reference_simulator.execute(reference_program).state

        for branch in result.branches:
            post_measurement_state = branch.state

            assert post_measurement_state is not None
            assert post_measurement_state.d == 2

            overlap = np.vdot(
                reference_state.state_vector, post_measurement_state.state_vector
            )

            assert np.isclose(np.abs(overlap) ** 2, 1.0)

    def test_scalar_phi_is_broadcast_to_all_measured_modes(self):
        """Scalar phi and an equal per-mode phi array must be equivalent."""
        shots = 20
        cutoff = 3
        phi = np.pi / 5

        def create_program(measurement_phi):
            with pq.Program() as program:
                pq.Q() | (
                    pq.NumberState([0, 0, 0]) + pq.NumberState([1, 0, 1])
                ) / np.sqrt(2)

                pq.Q(0, 1) | pq.HomodyneMeasurement(phi=measurement_phi)

            return program

        scalar_simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=cutoff, seed_sequence=123, hbar=1)
        )

        vector_simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=cutoff, seed_sequence=123, hbar=1)
        )

        scalar_result = scalar_simulator.execute(create_program(phi), shots=shots)

        vector_result = vector_simulator.execute(
            create_program(np.array([phi, phi])), shots=shots
        )

        assert np.allclose(scalar_result.samples, vector_result.samples)

        for scalar_branch, vector_branch in zip(
            scalar_result.branches, vector_result.branches
        ):
            scalar_state = scalar_branch.state
            vector_state = vector_branch.state

            assert scalar_state is not None
            assert vector_state is not None

            overlap = np.vdot(scalar_state.state_vector, vector_state.state_vector)

            assert np.isclose(np.abs(overlap) ** 2, 1.0)

    def test_measuring_all_modes_returns_no_post_measurement_state(self):
        shots = 10

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=4, seed_sequence=123, hbar=1)
        )

        with pq.Program() as program:
            pq.Q() | pq.NumberState([1, 0])

            pq.Q(0, 1) | pq.HomodyneMeasurement(phi=np.array([np.pi / 3, np.pi / 7]))

        result = simulator.execute(program, shots=shots)

        assert len(result.branches) == shots

        for branch in result.branches:
            assert branch.state is None

    def test_post_measurement_states_are_normalized(self):
        shots = 20

        simulator = pq.PureFockSimulator(
            d=3, config=pq.Config(cutoff=5, seed_sequence=123, hbar=1)
        )

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=0.2)
            pq.Q(1) | pq.Squeezing(r=0.1)

            pq.Q(0, 1) | pq.Beamsplitter5050()

            pq.Q(0, 1) | pq.HomodyneMeasurement(phi=np.array([np.pi / 3, np.pi / 7]))

        result = simulator.execute(program, shots=shots)

        for branch in result.branches:
            post_measurement_state = branch.state

            assert post_measurement_state is not None
            assert post_measurement_state.d == 1

            norm = np.vdot(
                post_measurement_state.state_vector,
                post_measurement_state.state_vector,
            ).real

            assert np.isclose(norm, 1.0)


def test_ParticleNumberMeasurement_resulting_state():
    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    with pq.Program() as program:
        pq.Q() | pq.NumberState([0, 1])

        pq.Q() | pq.Beamsplitter5050()

        pq.Q(0) | pq.ParticleNumberMeasurement()

    result = simulator.execute(program)

    assert result.state.d == 1
    assert np.isclose(sum(result.state.fock_probabilities), 1)


def test_ParticleNumberMeasurement_shots_None():
    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    p = 1 / np.pi

    with pq.Program() as program:
        pq.Q() | pq.NumberState([0, 1]) * np.sqrt(p)
        pq.Q() | pq.NumberState([1, 0]) * np.sqrt(1 - p)

        pq.Q(0) | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots=None)

    branches = result.branches
    assert len(branches) == 2

    assert branches[0].outcome == (0,)
    assert np.isclose(branches[0].frequency, p)

    assert branches[1].outcome == (1,)
    assert np.isclose(branches[1].frequency, 1 - p)


class TestMidCircuitMeasurements:
    """Test programs that contain mid-circuit measurements."""

    def test_multi_ParticleNumberMeasurement_in_one_program(self):
        simulator = pq.PureFockSimulator(
            d=4, config=pq.Config(cutoff=3, seed_sequence=123)
        )

        with pq.Program() as program:
            pq.Q() | pq.NumberState([0, 1, 1, 0])

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
            pq.Q() | pq.NumberState(
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
            pq.Q() | pq.NumberState([0, 0, 0, 2, 1, 1, 2]) * coeffs[0]
            pq.Q() | pq.NumberState([0, 0, 2, 0, 1, 1, 2]) * coeffs[1]
            pq.Q() | pq.NumberState([0, 1, 0, 1, 1, 1, 2]) * coeffs[2]
            pq.Q() | pq.NumberState([1, 1, 0, 1, 0, 1, 2]) * coeffs[3]
            pq.Q() | pq.NumberState([3, 0, 0, 0, 0, 1, 2]) * coeffs[4]

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
            pq.Q() | pq.NumberState([1, 1, res_samples[0], 0, res_samples[1]])
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
            pq.Q() | pq.NumberState([1, 1, 1, 0, 0])
            pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 1])
            pq.Q(2) | pq.Squeezing(0.0)
            pq.Q(measured_mode) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(d=5)
        with pytest.raises(ValueError, match=f"are not active: {{{measured_mode}}}"):
            simulator.execute(program, shots=1)

    def test_HomodyneMeasurement_followed_by_gate(self):
        shots = 10
        cutoff = 3
        hbar = 1.0
        homodyne_phi = np.pi / 5
        phaseshifter_phi = np.pi / 7

        with pq.Program() as program:
            pq.Q() | (pq.NumberState([0, 0]) + pq.NumberState([1, 1])) / np.sqrt(2)

            pq.Q(0) | pq.HomodyneMeasurement(phi=homodyne_phi)
            pq.Q(1) | pq.Phaseshifter(phi=phaseshifter_phi)

        simulator = pq.PureFockSimulator(
            d=2,
            config=pq.Config(cutoff=cutoff, hbar=hbar, seed_sequence=123),
        )

        result = simulator.execute(program, shots=shots)

        assert len(result.branches) == shots

        for branch in result.branches:
            assert branch.state is not None
            assert branch.state.d == 1

            q = branch.outcome[0] / np.sqrt(hbar)

            expected = np.zeros(cutoff, dtype=complex)
            expected[0] = 1.0
            expected[1] = (
                np.sqrt(2.0) * q * np.exp(1j * (phaseshifter_phi - homodyne_phi))
            )
            expected /= np.linalg.norm(expected)

            overlap = np.vdot(expected, branch.state.state_vector)

            assert np.isclose(np.abs(overlap) ** 2, 1.0)

    def test_HomodyneMeasurement_followed_by_ParticleNumberMeasurement(self):
        shots = 10

        with pq.Program() as program:
            pq.Q() | pq.NumberState([1, 0, 2, 1])

            pq.Q(1) | pq.HomodyneMeasurement(phi=np.pi / 3)
            pq.Q(3) | pq.Phaseshifter(phi=np.pi / 7)
            pq.Q(0, 2, 3) | pq.ParticleNumberMeasurement()

        simulator = pq.PureFockSimulator(
            d=4,
            config=pq.Config(cutoff=6, hbar=1.0, seed_sequence=123),
        )

        result = simulator.execute(program, shots=shots)

        assert len(result.samples) == shots

        for sample in result.samples:
            assert isinstance(sample[0], float)
            assert sample[1:] == (1, 2, 1)

        assert all(branch.state is None for branch in result.branches)

    def test_CV_gate_teleportation(self):
        cutoff = 16
        hbar = 2.0
        resource_squeezing = 0.9
        gate_s = 0.3
        input_displacement_r = 0.25
        input_displacement_phi = 0.2

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(
                r=input_displacement_r, phi=input_displacement_phi
            )

            # Prepare the finite-squeezing EPR resource on Alice's and Bob's modes.
            pq.Q(1) | pq.Squeezing(r=-resource_squeezing)
            pq.Q(2) | pq.Squeezing(r=resource_squeezing)
            pq.Q(1, 2) | pq.Beamsplitter5050()

            # Apply the gate to Bob's half of the resource before teleportation.
            pq.Q(2) | pq.QuadraticPhase(s=gate_s)

            # Alice performs the continuous-variable Bell measurement.
            pq.Q(0, 1) | pq.Beamsplitter5050()
            pq.Q(0) | pq.HomodyneMeasurement(phi=0.0)
            pq.Q(1) | pq.HomodyneMeasurement(phi=np.pi / 2)

            # Bob applies the gate-dependent feed-forward corrections.
            pq.Q(2) | pq.PositionDisplacement(
                x=lambda outcomes: outcomes[0] / np.sqrt(hbar)
            )
            pq.Q(2) | pq.MomentumDisplacement(
                p=lambda outcomes: (outcomes[1] + gate_s * outcomes[0]) / np.sqrt(hbar)
            )

        simulator = pq.PureFockSimulator(
            d=3,
            config=pq.Config(cutoff=cutoff, hbar=hbar, seed_sequence=16),
        )

        teleported_state = simulator.execute(program, shots=1).state

        with pq.Program() as reference_program:
            pq.Q() | pq.Vacuum()
            pq.Q(0) | pq.Displacement(
                r=input_displacement_r, phi=input_displacement_phi
            )
            pq.Q(0) | pq.QuadraticPhase(s=gate_s)

        reference_state = (
            pq.PureFockSimulator(
                d=1,
                config=pq.Config(cutoff=cutoff, hbar=hbar),
            )
            .execute(reference_program)
            .state
        )

        assert teleported_state is not None
        assert reference_state is not None
        assert reference_state.fidelity(teleported_state) > 0.99


def test_conditional_squeezing_with_function():
    r = 0.2

    program = pq.Program(
        instructions=[
            pq.NumberState([0, 2]) * np.sqrt(1 / 2),
            pq.NumberState([2, 0]) * np.sqrt(1 / 2),
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


def test_conditional_squeezing_with_expression():
    r = 0.2

    program = pq.Program(
        instructions=[
            pq.NumberState([0, 2]) * np.sqrt(1 / 2),
            pq.NumberState([2, 0]) * np.sqrt(1 / 2),
            pq.ParticleNumberMeasurement().on_modes(1),
            pq.Squeezing(r=r).on_modes(0).when("x[-1] == 2"),
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


def test_unresolved_squeezing_with_function():
    def f(x):
        return 0.01 * x[-1] ** 2

    cutoff = 7

    program = pq.Program(
        instructions=[
            pq.NumberState([0, 2]) * np.sqrt(1 / 3),
            pq.NumberState([1, 1]) * np.sqrt(1 / 3),
            pq.NumberState([2, 0]) * np.sqrt(1 / 3),
            pq.ParticleNumberMeasurement().on_modes(1),
            pq.Squeezing(r=f).on_modes(0),
        ]
    )

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=cutoff, seed_sequence=123)
    )

    result = simulator.execute(program, shots=10)

    for branch in result.branches:
        expected_state = (
            pq.PureFockSimulator(
                d=1, config=pq.Config(cutoff=cutoff - branch.outcome[0])
            )
            .execute_instructions(
                [
                    pq.NumberState([2 - branch.outcome[0]]),
                    pq.Squeezing(r=f(branch.outcome)),
                ]
            )
            .state
        )
        assert branch.state == expected_state


def test_unresolved_squeezing_with_expression():
    cutoff = 7

    program = pq.Program(
        instructions=[
            pq.NumberState([0, 2]) * np.sqrt(1 / 3),
            pq.NumberState([1, 1]) * np.sqrt(1 / 3),
            pq.NumberState([2, 0]) * np.sqrt(1 / 3),
            pq.ParticleNumberMeasurement().on_modes(1),
            pq.Squeezing(r="0.01 * x[-1] ** 2").on_modes(0),
        ]
    )

    simulator = pq.PureFockSimulator(
        d=2, config=pq.Config(cutoff=cutoff, seed_sequence=123)
    )

    result = simulator.execute(program, shots=10)

    for branch in result.branches:
        expected_state = (
            pq.PureFockSimulator(
                d=1, config=pq.Config(cutoff=cutoff - branch.outcome[0])
            )
            .execute_instructions(
                [
                    pq.NumberState([2 - branch.outcome[0]]),
                    pq.Squeezing(r=0.01 * branch.outcome[-1] ** 2),
                ]
            )
            .state
        )
        assert branch.state == expected_state
