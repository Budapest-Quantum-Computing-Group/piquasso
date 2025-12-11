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

import re

from scipy.stats import unitary_group

import piquasso as pq
from piquasso.api.exceptions import InvalidParameter


@pytest.fixture
def interferometer_matrix():
    return np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
    )


def test_sampling_raises_InvalidParameter_for_negative_shot_value(
    interferometer_matrix,
):
    invalid_shots = -1

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(InvalidParameter):
        simulator.execute(program, invalid_shots)


def test_sampling_samples_number(interferometer_matrix):
    shots = 100

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    assert len(result.samples) == shots, (
        f"Expected {shots} samples, " f"got: {len(program.result)}"
    )


def test_sampling_mode_permutation(interferometer_matrix):
    shots = 1

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    sample = result.samples[0]
    assert np.allclose(
        sample, [1, 0, 0, 1, 1]
    ), f"Expected [1, 0, 0, 1, 1], got: {sample}"


def test_sampling_multiple_samples_for_permutation_interferometer(
    interferometer_matrix,
):
    shots = 2

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q() | pq.Interferometer(interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots)

    samples = result.samples
    first_sample = samples[0]
    second_sample = samples[1]

    assert np.allclose(
        first_sample, second_sample
    ), f"Expected same samples, got: {first_sample} & {second_sample}"


basis_states = [
    (0, 1),
    (1, 0),
    (2, 0),
]


@pytest.mark.parametrize("input_state", basis_states)
def test_counts(input_state):
    shots = 100

    expected = {input_state: shots}
    with pq.Program() as program:
        pq.Q() | pq.StateVector(input_state)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=2)
    res = simulator.execute(program, shots=shots)
    counts = res.get_counts()
    assert len(counts) == len(expected)
    for r, e in zip(counts.items(), expected.items()):
        assert r[0] == e[0]
        assert np.isclose(r[1] / shots, e[1] / shots)


@pytest.mark.parametrize(
    "simulator, measurement_class",
    [
        (pq.PureFockSimulator, pq.HomodyneMeasurement),
        (pq.GaussianSimulator, pq.HeterodyneMeasurement),
    ],
)
def test_counts_raises(simulator, measurement_class):
    shots = 100

    with pq.Program() as program:
        if simulator == pq.PureFockSimulator:
            pq.Q() | pq.StateVector((0, 1, 0))
        else:
            pq.Q(0) | pq.Fourier()
            pq.Q(0, 2) | pq.Squeezing2(r=0.5, phi=0)
            pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0)
        pq.Q() | measurement_class()

    simulator = simulator(d=3)
    res = simulator.execute(program, shots=shots)
    with pytest.raises(
        NotImplementedError, match="method only supports samples that contain integers"
    ):
        res.get_counts()


def test_mach_zehnder():
    int_ = np.pi / 3
    ext = np.pi / 4

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q(0, 1) | pq.MachZehnder(int_=int_, ext=ext)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)


def test_fourier():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
        pq.Q(0) | pq.Fourier()
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)


def test_multiple_StateVector_with_ParticleNumberMeasurement_raises_error():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 1 / np.sqrt(2)
        pq.Q() | pq.StateVector([1, 1, 0, 1, 0]) * 1 / np.sqrt(2)

        pq.Q(0) | pq.Fourier()
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(pq.api.exceptions.NotImplementedCalculation) as error:
        simulator.execute(program, shots=1)

    assert error.value.args[0] == (
        "The instruction ParticleNumberMeasurement(modes=(0, 1, 2, 3, 4)) is "
        "not supported for states defined using multiple 'StateVector' instructions.\n"
        "If you need this feature to be implemented, please create an issue at "
        "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
    )


def test_uniform_loss():
    d = 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        for i in range(d):
            pq.Q(i) | pq.Loss(transmissivity=0.9)

    simulator = pq.SamplingSimulator(d=d)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy


def test_general_loss():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0) | pq.Loss(transmissivity=0.4)
        pq.Q(1) | pq.Loss(transmissivity=0.5)

        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    simulator.execute(program, shots=1)


def test_boson_sampling_seeded():
    seed_sequence = 123

    U = np.array(
        [
            [
                -0.17959207 - 0.29175972j,
                -0.64550941 - 0.02897055j,
                -0.023922 - 0.38854671j,
                -0.07908932 + 0.35617293j,
                -0.39779191 + 0.14902272j,
            ],
            [
                -0.36896208 + 0.19468375j,
                -0.11545557 + 0.20434514j,
                0.25548079 + 0.05220164j,
                -0.51002161 + 0.38442256j,
                0.48106678 - 0.25210091j,
            ],
            [
                0.25912844 + 0.16131742j,
                0.11886251 + 0.12632645j,
                0.69028213 - 0.25734432j,
                0.01276639 + 0.05841739j,
                0.03713264 + 0.57364845j,
            ],
            [
                -0.20314019 - 0.18973473j,
                0.59146854 + 0.28605532j,
                -0.11096495 - 0.26870144j,
                -0.47290354 - 0.0489408j,
                -0.42459838 + 0.01554643j,
            ],
            [
                -0.5021973 - 0.53474291j,
                0.24524545 - 0.0741398j,
                0.37786104 + 0.10225255j,
                0.46696955 + 0.10636677j,
                0.07171789 - 0.09194236j,
            ],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])

        pq.Q(all) | pq.Interferometer(U)

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (3, 1, 1, 0, 0),
        (2, 0, 0, 0, 3),
        (1, 0, 0, 0, 4),
        (0, 1, 0, 1, 3),
        (2, 0, 1, 0, 2),
        (0, 0, 1, 1, 3),
        (0, 1, 1, 2, 1),
        (1, 2, 0, 0, 2),
        (1, 0, 0, 3, 1),
        (1, 0, 4, 0, 0),
        (0, 0, 3, 1, 1),
        (2, 0, 3, 0, 0),
        (3, 0, 0, 0, 2),
        (0, 0, 4, 1, 0),
        (1, 1, 0, 1, 2),
        (0, 0, 2, 1, 2),
        (0, 2, 0, 1, 2),
        (0, 1, 0, 4, 0),
        (0, 1, 0, 3, 1),
        (0, 0, 1, 2, 2),
    ]

    assert samples == expected_samples


def test_LossyInterferometer_boson_sampling_seeded():
    seed_sequence = 123

    U = np.array(
        [
            [
                -0.17959207 - 0.29175972j,
                -0.64550941 - 0.02897055j,
                -0.023922 - 0.38854671j,
                -0.07908932 + 0.35617293j,
                -0.39779191 + 0.14902272j,
            ],
            [
                -0.36896208 + 0.19468375j,
                -0.11545557 + 0.20434514j,
                0.25548079 + 0.05220164j,
                -0.51002161 + 0.38442256j,
                0.48106678 - 0.25210091j,
            ],
            [
                0.25912844 + 0.16131742j,
                0.11886251 + 0.12632645j,
                0.69028213 - 0.25734432j,
                0.01276639 + 0.05841739j,
                0.03713264 + 0.57364845j,
            ],
            [
                -0.20314019 - 0.18973473j,
                0.59146854 + 0.28605532j,
                -0.11096495 - 0.26870144j,
                -0.47290354 - 0.0489408j,
                -0.42459838 + 0.01554643j,
            ],
            [
                -0.5021973 - 0.53474291j,
                0.24524545 - 0.0741398j,
                0.37786104 + 0.10225255j,
                0.46696955 + 0.10636677j,
                0.07171789 - 0.09194236j,
            ],
        ]
    )

    singular_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    lossy_interferometer_matrix = U @ np.diag(singular_values) @ U @ U.T

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])

        pq.Q(all) | pq.LossyInterferometer(lossy_interferometer_matrix)

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (1, 1, 1, 0, 0),
        (1, 0, 0, 1, 1),
        (1, 0, 0, 1, 0),
        (0, 0, 1, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 1, 1, 2),
        (1, 0, 1, 0, 2),
        (0, 0, 1, 0, 0),
        (1, 0, 0, 0, 3),
        (0, 1, 2, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 3, 0, 0),
        (0, 1, 0, 0, 1),
        (0, 1, 1, 0, 0),
        (0, 0, 2, 1, 0),
        (0, 0, 0, 1, 2),
        (0, 0, 1, 1, 0),
        (0, 2, 0, 0, 2),
        (0, 0, 1, 1, 0),
    ]

    assert samples == expected_samples


def test_LossyInterferometer_boson_sampling_uniform_losses():
    seed_sequence = 123

    U = np.array(
        [
            [
                -0.17959207 - 0.29175972j,
                -0.64550941 - 0.02897055j,
                -0.023922 - 0.38854671j,
                -0.07908932 + 0.35617293j,
                -0.39779191 + 0.14902272j,
            ],
            [
                -0.36896208 + 0.19468375j,
                -0.11545557 + 0.20434514j,
                0.25548079 + 0.05220164j,
                -0.51002161 + 0.38442256j,
                0.48106678 - 0.25210091j,
            ],
            [
                0.25912844 + 0.16131742j,
                0.11886251 + 0.12632645j,
                0.69028213 - 0.25734432j,
                0.01276639 + 0.05841739j,
                0.03713264 + 0.57364845j,
            ],
            [
                -0.20314019 - 0.18973473j,
                0.59146854 + 0.28605532j,
                -0.11096495 - 0.26870144j,
                -0.47290354 - 0.0489408j,
                -0.42459838 + 0.01554643j,
            ],
            [
                -0.5021973 - 0.53474291j,
                0.24524545 - 0.0741398j,
                0.37786104 + 0.10225255j,
                0.46696955 + 0.10636677j,
                0.07171789 - 0.09194236j,
            ],
        ]
    )

    singular_values = np.array([0.9] * 5)

    lossy_interferometer_matrix = U @ np.diag(singular_values) @ U @ U.T

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])

        pq.Q(all) | pq.LossyInterferometer(lossy_interferometer_matrix)

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (0, 2, 2, 0, 1),
        (1, 3, 0, 0, 1),
        (0, 2, 1, 0, 2),
        (0, 2, 1, 0, 2),
        (0, 1, 3, 0, 0),
        (2, 0, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 1, 1, 2, 1),
        (1, 0, 0, 0, 2),
        (1, 1, 0, 0, 3),
        (1, 1, 0, 0, 2),
        (5, 0, 0, 0, 0),
        (1, 0, 2, 0, 0),
        (0, 0, 4, 1, 0),
        (0, 1, 1, 0, 2),
        (2, 1, 1, 0, 1),
        (0, 1, 0, 0, 3),
        (0, 2, 1, 0, 2),
        (0, 3, 2, 0, 0),
        (1, 0, 0, 1, 1),
    ]

    assert samples == expected_samples


@pytest.mark.monkey
def test_post_select_random_unitary():
    d = 3

    interferometer_matrix = unitary_group.rvs(d)

    postselect_modes = (1, 2)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 0])

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

        pq.Q(*postselect_modes) | pq.PostSelectPhotons(photon_counts=(1, 0))

    simulator = pq.SamplingSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    state.normalize()
    state.validate()

    assert isinstance(state, pq.SamplingState)
    assert state.d == d - len(postselect_modes)


def test_PostSelectPhotons_state_vector():
    with pq.Program() as program_without_postselection:
        pq.Q() | pq.StateVector([1, 0, 1])
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

    with pq.Program() as program_with_postselection:
        pq.Q() | program_without_postselection
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 0])

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program_without_postselection).state
    postselected_state = simulator.execute(program_with_postselection).state

    assert state.d == 3
    assert postselected_state.d == 1

    state_vector_map = state.state_vector_map
    for occupation_number, coeff in postselected_state.state_vector_map.items():
        assert np.isclose(state_vector_map[(1, 0, occupation_number[0])], coeff)


@pytest.mark.monkey
def test_PostSelectPhotons_state_vector_random():
    interferometer_matrix = unitary_group.rvs(3, random_state=42)

    with pq.Program() as program_without_postselection:
        pq.Q(0, 1, 2) | pq.StateVector([1, 0, 1])
        pq.Q(all) | pq.Interferometer(interferometer_matrix)

    with pq.Program() as program_with_postselection:
        pq.Q() | program_without_postselection
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 0])

    simulator = pq.SamplingSimulator()
    state = simulator.execute(program_without_postselection).state
    postselected_state = simulator.execute(program_with_postselection).state

    state_vector_map = state.state_vector_map
    for occupation_number, coeff in postselected_state.state_vector_map.items():
        assert np.isclose(state_vector_map[(1, 0, occupation_number[0])], coeff)


def test_PostSelectPhotons_state_vector_with_more_photons():
    photon_counts = [2, 2]

    with pq.Program() as program_without_postselection:
        pq.Q(0, 1, 2, 3, 4) | pq.StateVector([1, 0, 1, 2, 1])
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

    with pq.Program() as program_with_postselection:
        pq.Q() | program_without_postselection
        pq.Q(1, 3) | pq.PostSelectPhotons(photon_counts)

    simulator = pq.SamplingSimulator(config=pq.Config(cutoff=6))
    postselected_state = simulator.execute(program_with_postselection).state
    state = simulator.execute(program_without_postselection).state

    postselected_state_vector_map = postselected_state.state_vector_map
    state_vector_map = state.state_vector_map
    for occupation_number, coeff in postselected_state_vector_map.items():
        index = (
            occupation_number[0],
            photon_counts[0],
            occupation_number[1],
            photon_counts[1],
            occupation_number[2],
        )
        assert np.isclose(state_vector_map[index], coeff)


@pytest.mark.monkey
def test_PostSelectPhotons_state_vector_with_more_photons_random():
    photon_counts = [2, 2]

    with pq.Program() as program_without_postselection:
        pq.Q(0, 1, 2, 3, 4) | pq.StateVector([1, 0, 1, 2, 1])

        pq.Q(all) | pq.Interferometer(unitary_group.rvs(5))

    with pq.Program() as program_with_postselection:
        pq.Q() | program_without_postselection
        pq.Q(1, 3) | pq.PostSelectPhotons(photon_counts)

    simulator = pq.SamplingSimulator(config=pq.Config(cutoff=6))
    postselected_state = simulator.execute(program_with_postselection).state
    state = simulator.execute(program_without_postselection).state

    postselected_state_vector_map = postselected_state.state_vector_map
    state_vector_map = state.state_vector_map
    for occupation_number, coeff in postselected_state_vector_map.items():
        index = (
            occupation_number[0],
            photon_counts[0],
            occupation_number[1],
            photon_counts[1],
            occupation_number[2],
        )
        assert np.isclose(state_vector_map[index], coeff)


def test_PostSelectPhotons_get_particle_detection_probability():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 0])

    simulator = pq.SamplingSimulator()
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability((1,))

    assert np.isclose(probability, 0.25)


@pytest.mark.monkey
def test_multiple_overlapping_PostSelectPhotons_raise_ValueError():
    interferometer_matrix = unitary_group.rvs(3, random_state=42)

    with pq.Program() as program:
        pq.Q(0, 1, 2) | pq.StateVector([2, 0, 1])
        pq.Q(all) | pq.Interferometer(interferometer_matrix)
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 0])
        pq.Q(1, 2) | pq.PostSelectPhotons(photon_counts=[0, 1])

    simulator = pq.SamplingSimulator()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Some modes of instruction "
            "PostSelectPhotons(photon_counts=[0, 1], modes=(1, 2)) "
            "are not active: {1}."
        ),
    ):
        simulator.execute(program)


@pytest.mark.monkey
def test_multiple_non_overlapping_PostSelectPhotons_no_error():
    interferometer_matrix = unitary_group.rvs(4, random_state=42)

    with pq.Program() as program_1:
        pq.Q(0, 1, 2, 3) | pq.StateVector([2, 0, 1, 1])
        pq.Q(all) | pq.Interferometer(interferometer_matrix)
        pq.Q(0) | pq.PostSelectPhotons(photon_counts=[1])
        pq.Q(1) | pq.PostSelectPhotons(photon_counts=[2])

    with pq.Program() as program_2:
        pq.Q(0, 1, 2, 3) | pq.StateVector([2, 0, 1, 1])
        pq.Q(all) | pq.Interferometer(interferometer_matrix)
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 2])

    simulator = pq.SamplingSimulator()

    state_1 = simulator.execute(program_1).state
    state_2 = simulator.execute(program_2).state

    assert state_1.d == state_2.d == 2
    assert state_1 == state_2


def test_PostSelectPhotons_no_infinite_loop_via_max_sample_generation_trials():
    with pq.Program() as program:
        pq.Q(0, 1, 2, 3) | pq.StateVector([2, 0, 2, 0])
        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[3, 0])
        pq.Q(2, 3) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(
        config=pq.Config(max_sample_generation_trials=10),
    )

    with pytest.raises(
        pq.api.exceptions.InvalidSimulation,
        match=(
            "Too many trials during sample generation. The specified postselection "
            "criteria may be very highly unlikely. Aborting. To increase the limit, "
            "set Config.max_sample_generation_trials to a higher value. "
            "Current value: 10."
        ),
    ):
        simulator.execute(program, shots=1)


class TestMidCircuitMeasurements:
    """Test programs that contain mid-circuit measurements."""

    @pytest.mark.monkey
    def test_ParticleNumberMeasurement_mid_circuit_not_allowed(self):
        """
        Test that an error is raised for mid-circuit measurements that are not allowed.
        """
        d = 3

        interferometer_matrix = unitary_group.rvs(d)
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
            pq.Q(2) | pq.ParticleNumberMeasurement()
            pq.Q(all) | pq.Interferometer(interferometer_matrix)

        simulator = pq.SamplingSimulator(d=5)
        with pytest.raises(
            pq.api.exceptions.InvalidSimulation,
            match="not allowed as a mid-circuit measurement",
        ):
            simulator.execute(program, shots=1)

    @pytest.mark.monkey
    def test_PostSelectPhotons_mid_circuit_allowed(self):
        """
        Test that no error is raised for mid-circuit PostSelectPhotons measurements.
        """

        interferometer_matrix = unitary_group.rvs(5, random_state=220)
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
            pq.Q(all) | pq.Interferometer(interferometer_matrix)
            pq.Q(0, 1) | pq.PostSelectPhotons(photon_counts=[1, 1])
            pq.Q(all) | pq.ParticleNumberMeasurement()

        simulator = pq.SamplingSimulator(d=5)
        samples = simulator.execute(program, shots=20).samples

        for sample in samples:
            assert sum(sample) == 1

    @pytest.mark.monkey
    def test_PostSelectPhotons_mid_circuit_bigger(self):
        input_state = np.array([2, 1, 3, 1, 1], dtype=int)

        d = len(input_state)

        unitary = unitary_group.rvs(d, random_state=320)

        program = pq.Program(
            instructions=[
                pq.StateVector(input_state),
                pq.Interferometer(unitary),
                pq.PostSelectPhotons(photon_counts=[1, 2]).on_modes(0, 2),
                pq.ParticleNumberMeasurement(),
            ]
        )

        simulator = pq.SamplingSimulator(d=d, config=pq.Config(seed_sequence=42))
        result = simulator.execute(program, shots=10)
        samples = result.samples

        for sample in samples:
            assert sum(sample) == sum(input_state) - 3

    @pytest.mark.monkey
    def test_PostSelectPhotons_mid_circuit_with_uniform_losses(self):
        input_state = np.array([1, 1, 1, 1], dtype=int)

        d = len(input_state)

        unitary = unitary_group.rvs(d, random_state=14)

        program = pq.Program(
            instructions=[
                pq.StateVector(input_state),
                pq.Interferometer(unitary),
                pq.LossyInterferometer(unitary @ np.diag([0.9] * d) @ unitary.conj().T),
                pq.PostSelectPhotons(photon_counts=[1, 1]).on_modes(0, 2),
                pq.ParticleNumberMeasurement(),
            ]
        )

        simulator = pq.SamplingSimulator(d=d, config=pq.Config(seed_sequence=42))
        result = simulator.execute(program, shots=10)
        samples = result.samples

        for sample in samples:
            assert sum(sample) <= sum(input_state) - 2

    def test_PostSelectPhotons_mid_circuit_with_generic_losses(self):
        input_state = np.array([1, 1, 1, 1], dtype=int)

        d = len(input_state)

        program = pq.Program(
            instructions=[
                pq.StateVector(input_state),
                pq.Beamsplitter5050().on_modes(0, 1),
                pq.Beamsplitter5050().on_modes(1, 2),
                pq.Beamsplitter5050().on_modes(2, 3),
                pq.Loss(0.9).on_modes(0),
                pq.Loss(0.8).on_modes(1),
                pq.Loss(0.7).on_modes(2),
                pq.Loss(0.6).on_modes(3),
                pq.PostSelectPhotons(photon_counts=[1, 1]).on_modes(0, 2),
                pq.ParticleNumberMeasurement(),
            ]
        )

        simulator = pq.SamplingSimulator(d=d, config=pq.Config(seed_sequence=42))
        result = simulator.execute(program, shots=10)
        samples = result.samples

        for sample in samples:
            assert sum(sample) <= sum(input_state) - 2

    def test_PostSelectPhotons_mid_circuit_invalid_modes_raises(self):
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 1, 0])
            pq.Q(0, 1) | pq.Beamsplitter5050()
            pq.Q(1, 2) | pq.Beamsplitter5050()
            pq.Q(0) | pq.PostSelectPhotons(photon_counts=[1])
            pq.Q(0, 1) | pq.Beamsplitter5050()

        simulator = pq.SamplingSimulator()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Some modes of instruction Beamsplitter5050(modes=(0, 1)) "
                "are not active: {0}."
            ),
        ):
            simulator.execute(program, shots=1)

    def test_PostSelectPhotons_deferred_measurement_principle(self):
        with pq.Program() as program_1:
            pq.Q() | pq.StateVector([1, 1, 0])
            pq.Q(0, 1) | pq.Beamsplitter5050()
            pq.Q(1, 2) | pq.Beamsplitter5050()
            pq.Q(0) | pq.PostSelectPhotons(photon_counts=[1])

        with pq.Program() as program_2:
            pq.Q() | pq.StateVector([1, 1, 0])
            pq.Q(0, 1) | pq.Beamsplitter5050()
            pq.Q(0) | pq.PostSelectPhotons(photon_counts=[1])
            pq.Q(1, 2) | pq.Beamsplitter5050()

        simulator = pq.SamplingSimulator()
        state_1 = simulator.execute(program_1).state
        state_2 = simulator.execute(program_2).state

        assert state_1 == state_2

        assert np.allclose(state_1.state_vector, state_2.state_vector)
