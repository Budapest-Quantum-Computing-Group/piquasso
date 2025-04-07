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

import pytest

import numpy as np

import piquasso as pq

from piquasso.fermionic._utils import binary_to_fock_indices

for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@pytest.mark.monkey
@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_equivalence_1_particle(
    connector,
    generate_unitary_matrix,
):
    d = 3

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0, 1])

        pq.Q() | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix
    density_matrix_fock = fock_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(
        density_matrix_fock[1 : (d + 1), 1 : (d + 1)],
        U @ np.diag([0, 0, 1]) @ U.conj().T,
    )

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@pytest.mark.monkey
@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_equivalence_2_particles(
    connector,
    generate_unitary_matrix,
):
    d = 3

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1])

        pq.Q() | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_fock = fock_simulator.execute(program).state.density_matrix
    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@pytest.mark.monkey
@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_equivalence_3_particles(
    connector,
    generate_unitary_matrix,
):
    d = 4

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0, 1, 1])

        pq.Q() | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_fock = fock_simulator.execute(program).state.density_matrix
    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@pytest.mark.monkey
@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_equivalence_n_particles_random(
    connector,
    generate_unitary_matrix,
):
    d = np.random.randint(1, 6)

    occupation_numbers = np.random.randint(0, 2, d)

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers)

        pq.Q() | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_fock = fock_simulator.execute(program).state.density_matrix
    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@pytest.mark.monkey
@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_equivalence_subsystem_random(
    connector,
    generate_unitary_matrix,
):
    d = 3

    occupation_numbers = [1, 1, 0]

    modes = (0, 1)

    U = generate_unitary_matrix(2)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(occupation_numbers)

        pq.Q(*modes) | pq.Interferometer(U)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_fock = fock_simulator.execute(program).state.density_matrix
    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_squeezing2_equivalence(connector):
    d = 2

    r = 0.2
    phi = np.pi / 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1])

        pq.Q(0, 1) | pq.Squeezing2(r=r, phi=phi)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix
    density_matrix_fock = fock_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)


@for_all_connectors
def test_PureFockSimulator_GaussianSimulator_IsingXX_equivalence(connector):
    d = 2

    phi = np.pi / 5

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1])

        pq.Q(0, 1) | pq.fermionic.IsingXX(phi=phi)

    fock_simulator = pq.fermionic.PureFockSimulator(
        d=d, config=pq.Config(cutoff=d + 1), connector=connector
    )
    gaussian_simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    density_matrix_gaussian = gaussian_simulator.execute(program).state.density_matrix
    density_matrix_fock = fock_simulator.execute(program).state.density_matrix

    indices = binary_to_fock_indices(d)

    density_matrix_gaussian_reordered = density_matrix_gaussian[
        np.ix_(indices, indices)
    ]

    assert np.allclose(density_matrix_fock, density_matrix_gaussian_reordered)
