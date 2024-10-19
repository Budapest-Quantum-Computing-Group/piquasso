#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

        pq.Q() | pq.ParticleNumberMeasurement()

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
        (3, 0, 0, 0, 2),
        (0, 0, 1, 2, 2),
        (0, 1, 0, 4, 0),
        (1, 0, 0, 0, 4),
        (0, 1, 0, 1, 3),
        (1, 0, 0, 3, 1),
        (2, 0, 3, 0, 0),
        (3, 1, 1, 0, 0),
        (0, 1, 0, 3, 1),
        (1, 2, 0, 0, 2),
        (1, 0, 4, 0, 0),
        (0, 1, 1, 2, 1),
        (0, 0, 1, 1, 3),
        (0, 2, 0, 1, 2),
        (2, 0, 1, 0, 2),
        (0, 0, 3, 1, 1),
        (0, 0, 4, 1, 0),
        (2, 0, 0, 0, 3),
        (0, 0, 2, 1, 2),
        (1, 1, 0, 1, 2),
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
        (0, 0, 3, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 1, 0),
        (1, 0, 0, 1, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0),
        (1, 1, 1, 0, 0),
        (0, 2, 0, 0, 2),
        (0, 0, 1, 0, 0),
        (1, 0, 1, 0, 2),
        (1, 0, 0, 0, 3),
        (0, 0, 1, 1, 2),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 1, 2),
        (0, 1, 1, 1, 0),
        (0, 1, 2, 0, 0),
        (0, 1, 0, 0, 1),
        (1, 0, 0, 1, 1),
        (0, 0, 2, 1, 0),
        (0, 1, 1, 0, 0),
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
        (1, 0, 2, 0, 0),
        (1, 0, 0, 1, 1),
        (0, 2, 1, 0, 2),
        (1, 0, 0, 0, 2),
        (5, 0, 0, 0, 0),
        (0, 2, 2, 0, 1),
        (0, 3, 2, 0, 0),
        (0, 1, 1, 2, 1),
        (1, 1, 0, 0, 3),
        (0, 0, 0, 2, 2),
        (0, 2, 1, 0, 2),
        (2, 0, 2, 0, 0),
        (0, 1, 0, 0, 3),
        (0, 1, 3, 0, 0),
        (1, 1, 0, 0, 2),
        (0, 0, 4, 1, 0),
        (1, 3, 0, 0, 1),
        (2, 1, 1, 0, 1),
        (0, 2, 1, 0, 2),
        (0, 1, 1, 0, 2),
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

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=postselect_modes, photon_counts=(1, 0)
        )

    simulator = pq.SamplingSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    state.normalize()
    state.validate()

    assert state.d == d - len(postselect_modes)
