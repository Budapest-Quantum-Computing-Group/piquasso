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
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])
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
        pq.Q() | pq.StateVector(initial_occupation)
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
        pq.Q() | pq.StateVector(occupation)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.GaussianSimulator(
        d=d, connector=connector, config=pq.Config(seed_sequence=42)
    )

    samples = simulator.execute(program, shots=10).samples

    for sample in samples:
        assert sample == tuple(occupation)


def test_particle_number_measurement_reproduces_exact_distribution():
    d = 4

    U = unitary_group.rvs(d, random_state=7)

    with pq.Program() as state_program:
        pq.Q() | pq.StateVector([1, 0, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    state_simulator = pq.fermionic.GaussianSimulator(d=d)
    state = state_simulator.execute(state_program).state

    with np.errstate(invalid="ignore"):
        exact_probabilities = np.nan_to_num(state.fock_probabilities, nan=0.0)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1, 0])
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    shots = 20000

    simulator = pq.fermionic.GaussianSimulator(d=d, config=pq.Config(seed_sequence=42))
    samples = simulator.execute(program, shots=shots).samples

    empirical_probabilities = np.zeros(2**d)
    for sample in samples:
        index = int("".join(map(str, sample)), 2)
        empirical_probabilities[index] += 1
    empirical_probabilities /= shots

    assert np.allclose(empirical_probabilities, exact_probabilities, atol=1e-2)


def test_particle_number_measurement_mean_particle_numbers():
    d = 5

    U = unitary_group.rvs(d, random_state=11)

    with pq.Program() as state_program:
        pq.Q() | pq.StateVector([1, 1, 0, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    state_simulator = pq.fermionic.GaussianSimulator(d=d)
    state = state_simulator.execute(state_program).state

    exact_mean = state.mean_particle_numbers(modes=tuple(range(d)))

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0, 1, 0])
        pq.Q(all) | pq.Interferometer(U)
        pq.Q() | pq.ParticleNumberMeasurement()

    shots = 20000

    simulator = pq.fermionic.GaussianSimulator(d=d, config=pq.Config(seed_sequence=42))
    samples = simulator.execute(program, shots=shots).samples

    empirical_mean = np.array(samples).mean(axis=0)

    assert np.allclose(empirical_mean, exact_mean, atol=2e-2)
