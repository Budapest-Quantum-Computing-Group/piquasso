#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

from scipy.linalg import polar, sinhm, coshm, expm


def is_proportional(first, second):
    first = np.array(first)
    second = np.array(second)

    index = np.argmax(first)

    proportion = first[index] / second[index]

    return np.allclose(first, proportion * second)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_should_be_numpy_array_of_floats(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.dtype == np.float64


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_squeezed_state(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99502075,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00494212,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


def test_density_matrix_with_squeezed_state():
    d = 2

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

    gaussian_simulator = pq.GaussianSimulator(d=d)
    gaussian_state = gaussian_simulator.execute(gaussian_program).state

    gaussian_density_matrix = gaussian_state.density_matrix

    normalization = 1 / sum(np.diag(gaussian_density_matrix))

    with pq.Program() as fock_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

    fock_simulator = pq.FockSimulator(d=d)
    fock_state = fock_simulator.execute(fock_program).state

    fock_density_matrix = fock_state.density_matrix

    assert np.allclose(normalization * gaussian_density_matrix, fock_density_matrix)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_displaced_state(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0.0,
            0.0,
            0.03368973,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.08422434,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1403739,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_displaced_state_with_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0.0,
            0.0252673,
            0.00842243,
            0.0,
            0.0,
            0.04737619,
            0.0,
            0.03158413,
            0.00526402,
            0.0,
            0.0,
            0.0,
            0.05922024,
            0.0,
            0.0,
            0.05922024,
            0.0,
            0.01974008,
            0.00219334,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_squeezed_state_with_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99502075,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00277994,
            0.0,
            0.0018533,
            0.00030888,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_two_single_mode_squeezings(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)
        pq.Q(1) | pq.Squeezing(r=0.2, phi=0.7)

    simulator = SimulatorClass(d=2)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [0.9754467, 0.0, 0.0, 0.01900025, 0.0, 0.0048449, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_two_mode_squeezing(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99006629,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00983503,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_two_mode_squeezing_and_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99006629,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00368814,
            0.0,
            0.00245876,
            0.00368814,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_quadratic_phase(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.QuadraticPhase(s=0.4)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.98058068,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01885732,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_position_displacement(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.PositionDisplacement(x=0.2)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.96078944,
        0.0,
        0.0,
        0.03843158,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00076863,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00001025,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_momentum_displacement(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.MomentumDisplacement(p=0.2)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.96078944,
        0.0,
        0.0,
        0.03843158,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00076863,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00001025,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_position_displacement_is_HBAR_independent(
    SimulatorClass,
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.PositionDisplacement(x=0.4)

    simulator1 = SimulatorClass(d=3, config=pq.Config(hbar=2))
    simulator2 = SimulatorClass(d=3, config=pq.Config(hbar=42))

    state1 = simulator1.execute(program).state
    state2 = simulator2.execute(program).state

    probabilities1 = state1.fock_probabilities
    probabilities2 = state2.fock_probabilities

    assert np.allclose(probabilities1, probabilities2)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_momentum_displacement_is_HBAR_independent(
    SimulatorClass,
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.MomentumDisplacement(p=0.4)

    simulator1 = SimulatorClass(d=3, config=pq.Config(hbar=2))
    simulator2 = SimulatorClass(d=3, config=pq.Config(hbar=42))

    state1 = simulator1.execute(program).state
    state2 = simulator2.execute(program).state

    probabilities1 = state1.fock_probabilities
    probabilities2 = state2.fock_probabilities

    assert np.allclose(probabilities1, probabilities2)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_general_gaussian_transform(SimulatorClass):
    squeezing_matrix = np.array(
        [
            [0.1, 0.2 + 0.3j],
            [0.2 + 0.3j, 0.1],
        ],
        dtype=complex,
    )

    rotation_matrix = np.array(
        [
            [1, 3 - 2j],
            [3 + 2j, 1],
        ],
        dtype=complex,
    )

    U, r = polar(squeezing_matrix)

    passive = expm(-1j * rotation_matrix) @ coshm(r)
    active = expm(-1j * rotation_matrix) @ U @ sinhm(r)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.GaussianTransform(passive=passive, active=active)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.864652,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05073686,
        0.0,
        0.02118922,
        0.0379305,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.monkey
def test_monkey_fock_probabilities_with_general_gaussian_transform(
    generate_unitary_matrix, generate_complex_symmetric_matrix
):
    d = 3

    squeezing_matrix = generate_complex_symmetric_matrix(3)
    U, r = polar(squeezing_matrix)

    global_phase = generate_unitary_matrix(d)
    passive = global_phase @ coshm(r)
    active = global_phase @ sinhm(r) @ U.conj()

    with pq.Program() as fock_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    fock_simulator = pq.FockSimulator(d=d)
    fock_state = fock_simulator.execute(fock_program).state

    fock_representation_probabilities = fock_state.fock_probabilities

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    gaussian_simulator = pq.GaussianSimulator(d=d)
    gaussian_state = gaussian_simulator.execute(gaussian_program).state

    gaussian_representation_probabilities = gaussian_state.fock_probabilities

    normalization = 1 / sum(gaussian_representation_probabilities)

    assert np.allclose(
        fock_representation_probabilities,
        normalization * gaussian_representation_probabilities,
    )


@pytest.mark.monkey
def test_monkey_get_density_matrix_with_general_gaussian_transform(
    generate_unitary_matrix, generate_complex_symmetric_matrix
):
    d = 3

    squeezing_matrix = generate_complex_symmetric_matrix(3)
    U, r = polar(squeezing_matrix)

    global_phase = generate_unitary_matrix(d)
    passive = global_phase @ coshm(r)
    active = global_phase @ sinhm(r) @ U.conj()

    with pq.Program() as fock_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    fock_simulator = pq.FockSimulator(d=d)
    fock_state = fock_simulator.execute(fock_program).state

    fock_representation_probabilities = fock_state.fock_probabilities

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    gaussian_simulator = pq.GaussianSimulator(d=d)
    gaussian_state = gaussian_simulator.execute(gaussian_program).state

    gaussian_representation_probabilities = gaussian_state.fock_probabilities

    normalization = 1 / sum(gaussian_representation_probabilities)

    assert np.allclose(
        fock_representation_probabilities,
        normalization * gaussian_representation_probabilities,
    )


def test_sampling_backend_equivalence_for_two_mode_beamsplitter():
    initial_occupation_numbers = (1, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(
        fock_state.fock_probabilities,
        sampling_state.fock_probabilities,
    )


def test_sampling_backend_equivalence_complex_scenario():
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter(np.pi / 4)

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(fock_state.fock_probabilities, sampling_state.fock_probabilities)


@pytest.mark.monkey
def test_sampling_backend_equivalence_with_random_interferometer(
    generate_unitary_matrix,
):
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    interferometer_matrix = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(
        fock_state.fock_probabilities,
        sampling_state.fock_probabilities,
    )


def test_wigner_function_equivalence():
    config = pq.Config(cutoff=10, hbar=42)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=0.10 - 0.05j)
        pq.Q(0) | pq.Squeezing(r=0.10)

    fock_simulator = pq.FockSimulator(d=1, config=config)
    fock_state = fock_simulator.execute(program).state

    fock_wigner_function_values = fock_state.wigner_function(
        positions=[0.10, 0.11],
        momentums=[-0.05, -0.06],
    )

    gaussian_simulator = pq.GaussianSimulator(d=1, config=config)
    gaussian_state = gaussian_simulator.execute(program).state

    gaussian_wigner_function_values = gaussian_state.wigner_function(
        positions=[[0.10], [0.11]],
        momentums=[[-0.05], [-0.06]],
    )

    assert np.allclose(fock_wigner_function_values, gaussian_wigner_function_values)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fidelity(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.2, phi=-np.pi / 3)
        pq.Q(0) | pq.Displacement(r=-0.1, phi=0)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.2, phi=np.pi / 3)
        pq.Q(0) | pq.Displacement(r=-0.1, phi=0)

    generic_simulator = SimulatorClass(d=1, config=pq.Config(cutoff=10))

    state_1 = generic_simulator.execute(program_1).state
    state_2 = generic_simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 0.9421652615828)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_cubic_phase_equivalency(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.CubicPhase(gamma=0.1)
        pq.Q(1) | pq.CubicPhase(gamma=-0.07)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.CubicPhase(gamma=0.1)
        pq.Q(1) | pq.CubicPhase(gamma=-0.07)

    generic_simulator = SimulatorClass(d=2, config=pq.Config(cutoff=10))

    state_1 = generic_simulator.execute(program_1).state
    state_2 = generic_simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 1.0)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_kerr_equivalency(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.05)
        pq.Q(0, 1) | pq.Kerr(xi=[-1, 1])

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.05)
        pq.Q(all) | pq.Kerr(xi=[-1, 1])

    generic_simulator = SimulatorClass(d=2, config=pq.Config(cutoff=10))

    state_1 = generic_simulator.execute(program_1).state
    state_2 = generic_simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 1.0)
    assert np.isclose(fidelity, state_2.fidelity(state_1))
