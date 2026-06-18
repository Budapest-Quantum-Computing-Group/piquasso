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

from scipy.stats import unitary_group

for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_particle_number_measurement_samples_are_binary(connector):
    d = 5

    U = unitary_group.rvs(d, random_state=123)

    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=100).samples

    assert len(samples) == 100

    for sample in samples:
        assert len(sample) == d
        assert all(occupation in (0, 1) for occupation in sample)


@for_all_connectors
def test_particle_number_measurement_conserves_particle_number(connector):
    d = 5

    U = unitary_group.rvs(d, random_state=321)

    initial_occupation = [1, 1, 1, 0, 0]

    with pq.Program() as program:
        pq.Q() | pq.NumberState(initial_occupation)
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=100).samples

    expected_particle_number = sum(initial_occupation)

    for sample in samples:
        assert sum(sample) == expected_particle_number


@for_all_connectors
def test_particle_number_measurement_on_vacuum_yields_zeros(connector):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=10).samples

    for sample in samples:
        assert sample == (0, 0, 0)


@for_all_connectors
def test_particle_number_measurement_on_occupation_number_state(connector):
    d = 4

    occupation = [1, 0, 1, 1]

    with pq.Program() as program:
        pq.Q() | pq.NumberState(occupation)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=10).samples

    for sample in samples:
        assert sample == tuple(occupation)


def test_particle_number_measurement_is_seed_reproducible():
    d = 5

    U = unitary_group.rvs(d, random_state=99)

    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator1 = pq.fermionic.GaussianSimulator(d=d, config=pq.Config(seed_sequence=42))
    simulator2 = pq.fermionic.GaussianSimulator(d=d, config=pq.Config(seed_sequence=42))

    samples1 = simulator1.execute(program, shots=20).samples
    samples2 = simulator2.execute(program, shots=20).samples

    assert samples1 == samples2


def test_particle_number_measurement_samples_are_connector_independent():
    d = 5

    U = unitary_group.rvs(d, random_state=99)

    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    numpy_simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=pq.NumpyConnector(), config=pq.Config(seed_sequence=42)
    )
    jax_simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=pq.JaxConnector(), config=pq.Config(seed_sequence=42)
    )

    numpy_samples = numpy_simulator.execute(program, shots=20).samples
    jax_samples = jax_simulator.execute(program, shots=20).samples

    assert numpy_samples == jax_samples


@for_all_connectors
def test_particle_number_measurement_on_identity_interferometer(connector):
    d = 4

    occupation = [1, 0, 1, 0]

    with pq.Program() as program:
        pq.Q() | pq.NumberState(occupation)
        pq.Q(all) | pq.Interferometer(np.identity(d, dtype=complex))
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=10).samples

    for sample in samples:
        assert tuple(sample) == tuple(occupation)


@for_all_connectors
def test_particle_number_measurement_on_permutation_interferometer(connector):
    d = 4

    occupation = [1, 1, 0, 0]

    # A cyclic permutation of the modes.
    permutation = np.array(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q() | pq.NumberState(occupation)
        pq.Q(all) | pq.Interferometer(permutation)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=10).samples

    # A permutation interferometer maps a Fock basis state to another Fock basis
    # state, so the outcome is deterministic and a permutation of the input.
    for sample in samples:
        assert sample == samples[0]
        assert sorted(sample) == sorted(occupation)


@for_all_connectors
def test_particle_number_measurement_conserves_particles_within_blocks(connector):
    d = 4

    occupation = [1, 0, 1, 0]

    beamsplitter = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    zeros = np.zeros((2, 2), dtype=complex)
    interferometer = np.block([[beamsplitter, zeros], [zeros, beamsplitter]])

    with pq.Program() as program:
        pq.Q() | pq.NumberState(occupation)
        pq.Q(all) | pq.Interferometer(interferometer)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=20).samples

    # The interferometer does not couple modes {0, 1} to modes {2, 3}, so the
    # particle number is conserved within each block for every shot.
    for sample in samples:
        assert sample[0] + sample[1] == 1
        assert sample[2] + sample[3] == 1
