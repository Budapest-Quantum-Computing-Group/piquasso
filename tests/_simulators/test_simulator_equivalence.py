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

import pytest
import numpy as np
import piquasso as pq
import tensorflow as tf

import collections

from functools import partial

from scipy.stats import unitary_group
from scipy.linalg import polar, sinhm, coshm, expm


def is_proportional(first, second, rtol=1e-5):
    first = np.array(first)
    second = np.array(second)

    index = np.argmax(first)

    proportion = first[index] / second[index]

    return np.allclose(first, proportion * second, rtol=rtol)


tf_purefock_simulators = (
    partial(
        pq.PureFockSimulator,
        connector=pq.TensorflowConnector(),
    ),
    partial(
        pq.PureFockSimulator,
        connector=pq.TensorflowConnector(decorate_with=tf.function),
    ),
)

jax_purefock_simulator = [
    partial(
        pq.PureFockSimulator,
        connector=pq.JaxConnector(),
    ),
]


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
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

    assert isinstance(probabilities, collections.abc.Iterable)
    assert probabilities.dtype == np.float64


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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

    with pq.Program() as fock_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

    fock_simulator = pq.FockSimulator(d=d)
    fock_state = fock_simulator.execute(fock_program).state

    fock_density_matrix = fock_state.density_matrix

    assert np.allclose(gaussian_density_matrix, fock_density_matrix)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_displaced_state(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.sqrt(5), phi=np.angle(1 + 2j))

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0.03368973,
            0.0,
            0.0,
            0.08422434,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1403739,
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
        pq.PureFockSimulator,
        pq.FockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.GaussianSimulator,
    ),
)
def test_Displacement_equivalence_on_multiple_modes(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.01)
        pq.Q(1) | pq.Displacement(r=0.02)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

    simulator = SimulatorClass(d=2)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [
            0.99950012,
            0.00020242,
            0.00029733,
            0.00000002,
            0.00000006,
            0.00000004,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_displaced_state_with_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.sqrt(5), phi=np.angle(1 + 2j))
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0.00842243,
            0.0252673,
            0.0,
            0.00526402,
            0.03158413,
            0.0,
            0.04737619,
            0.0,
            0.0,
            0.00219334,
            0.01974008,
            0.0,
            0.05922024,
            0.0,
            0.0,
            0.05922024,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_squeezed_state_with_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

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
            0.00030888,
            0.0018533,
            0.0,
            0.00277994,
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
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        [0.9754467, 0.0, 0.0, 0.0048449, 0.0, 0.01900025, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.FockSimulator,
    ),
)
def test_Squeezing_equivalence_on_multiple_modes(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 5)
        pq.Q(1) | pq.Squeezing(r=0.2, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

    simulator = SimulatorClass(d=2)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [0.97613795, 0.0, 0.0, 0.01246268, 0.00601937, 0.00537999, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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

    # NOTE: It should not happen that the probabilities are negative, but in some cases
    # (possibly due to floating point errors) they turn out to be slightly negative.
    assert all(
        probability >= 0.0 or np.isclose(probability, 0.0)
        for probability in probabilities
    )
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99006629,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
        pq.FockSimulator,
    ),
)
def test_fock_probabilities_with_two_mode_squeezing_and_beamsplitter(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

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
            0.00368814,
            0.00245876,
            0.0,
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
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        0.03843158,
        0.0,
        0.0,
        0.00076863,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00001025,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        0.03843158,
        0.0,
        0.0,
        0.00076863,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00001025,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        *tf_purefock_simulators,
        *jax_purefock_simulator,
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
        0.88726976,
        0.0,
        0.0,
        0.0,
        0.03892269,
        0.02174349,
        0.0,
        0.05206405,
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
        0.0,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities, rtol=1e-4)


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

    assert np.allclose(
        fock_representation_probabilities,
        gaussian_representation_probabilities,
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

    assert np.allclose(
        fock_representation_probabilities,
        gaussian_representation_probabilities,
    )


def test_sampling_backend_equivalence_for_two_mode_beamsplitter():
    initial_occupation_numbers = (1, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

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


def test_sampling_backend_equivalence_two_mode_Beamsplitter5050_fock_probabilities():
    initial_occupation_numbers = (1, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter5050()

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


def test_sampling_backend_equivalence_two_mode_Beamsplitter5050_state_vector():
    initial_occupation_numbers = (1, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter5050()

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(
        fock_state.state_vector,
        sampling_state.state_vector,
    )


def test_sampling_backend_equivalence_complex_scenario_fock_probabilities():
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(fock_state.fock_probabilities, sampling_state.fock_probabilities)


def test_sampling_backend_equivalence_complex_scenario_state_vector():
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

    config = pq.Config(cutoff=sum(initial_occupation_numbers) + 1)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(fock_state.state_vector, sampling_state.state_vector)


def test_sampling_backend_equivalence_complex_scenario_fock_probabilities_2():
    initial_occupation_numbers = np.array(
        [
            [1, 1, 0, 1],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])
    d = initial_occupation_numbers.shape[1]

    with pq.Program() as program:
        for idx in range(len(coefficients)):
            pq.Q() | pq.StateVector(
                initial_occupation_numbers[idx], coefficient=coefficients[idx]
            )

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

    config = pq.Config(cutoff=4)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(fock_state.fock_probabilities, sampling_state.fock_probabilities)


def test_sampling_backend_equivalence_complex_scenario_state_vector_2():
    initial_occupation_numbers = np.array(
        [
            [1, 1, 0, 1],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])
    d = initial_occupation_numbers.shape[1]

    with pq.Program() as program:
        for idx in range(len(coefficients)):
            pq.Q() | pq.StateVector(
                initial_occupation_numbers[idx], coefficient=coefficients[idx]
            )

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

    config = pq.Config(cutoff=4)

    fock_simulator = pq.PureFockSimulator(d=d, config=config)
    fock_state = fock_simulator.execute(program).state
    fock_state.validate()

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(fock_state.state_vector, sampling_state.state_vector)


@pytest.mark.monkey
def test_sampling_backend_equivalence_with_random_interferometer_fock_probabilities(
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


@pytest.mark.monkey
def test_sampling_backend_equivalence_with_random_interferometer_state_vector(
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
        fock_state.state_vector,
        sampling_state.state_vector,
    )


def test_wigner_function_equivalence():
    config = pq.Config(cutoff=10, hbar=42)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.10, phi=np.angle(1 - 0.5j))
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
def test_fidelity_for_1_mode(SimulatorClass):
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
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fidelity_for_nondisplaced_states_on_2_modes(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(0.1)
        pq.Q(1) | pq.Squeezing(0.2)

        pq.Q(all) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 7)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(0.3)
        pq.Q(1) | pq.Squeezing(0.1)

        pq.Q(all) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 9)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=9))

    state_1 = simulator.execute(program_1).state
    state_2 = simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 0.9735085877314046)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fidelity_for_displaced_states_on_2_modes(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.4)
        pq.Q(1) | pq.Displacement(0.5)

        pq.Q(0) | pq.Squeezing(0.1)
        pq.Q(1) | pq.Squeezing(0.2)

        pq.Q(all) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 7)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.5)
        pq.Q(1) | pq.Displacement(0.4)

        pq.Q(0) | pq.Squeezing(0.3)
        pq.Q(1) | pq.Squeezing(0.1)

        pq.Q(all) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 9)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=10))

    state_1 = simulator.execute(program_1).state
    state_2 = simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 0.9346675071279842)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fidelity_for_nondisplaced_pure_states_on_3_modes(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.04)
        pq.Q(1) | pq.Displacement(0.05)
        pq.Q(2) | pq.Displacement(0.1)

        pq.Q(0) | pq.Squeezing(0.01)
        pq.Q(1) | pq.Squeezing(0.02)
        pq.Q(2) | pq.Squeezing(0.03)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 7)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 9)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.05)
        pq.Q(1) | pq.Displacement(0.04)
        pq.Q(2) | pq.Displacement(0.02)

        pq.Q(0) | pq.Squeezing(0.03)
        pq.Q(1) | pq.Squeezing(0.01)
        pq.Q(2) | pq.Squeezing(0.02)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=np.pi / 9)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 3)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=7))

    state_1 = simulator.execute(program_1).state
    state_2 = simulator.execute(program_2).state
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 0.9889124929545777)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_fidelity_for_nondisplaced_mixed_states_on_3_modes(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.04)
        pq.Q(1) | pq.Displacement(0.05)
        pq.Q(2) | pq.Displacement(0.1)

        pq.Q(0) | pq.Squeezing(0.01)
        pq.Q(1) | pq.Squeezing(0.02)
        pq.Q(2) | pq.Squeezing(0.03)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 7)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 9)

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(0.05)
        pq.Q(1) | pq.Displacement(0.04)
        pq.Q(2) | pq.Displacement(0.02)

        pq.Q(0) | pq.Squeezing(0.03)
        pq.Q(1) | pq.Squeezing(0.01)
        pq.Q(2) | pq.Squeezing(0.02)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 2, phi=np.pi / 9)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 3)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=7))

    state_1 = simulator.execute(program_1).state.reduced(modes=(0, 1))
    state_2 = simulator.execute(program_2).state.reduced(modes=(0, 1))
    fidelity = state_1.fidelity(state_2)

    assert np.isclose(fidelity, 0.9959871270027937)
    assert np.isclose(fidelity, state_2.fidelity(state_1))


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        *tf_purefock_simulators,
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
        *tf_purefock_simulators,
    ),
)
def test_CubicPhase_equivalence_on_multiple_modes(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.CubicPhase(gamma=0.1)
        pq.Q(1) | pq.CubicPhase(gamma=0.2)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

    simulator = SimulatorClass(d=2)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [
            0.97960898,
            0.00474935,
            0.00710543,
            0.00029005,
            0.00030477,
            0.0001192,
            0.00306916,
            0.00174448,
            0.00261566,
            0.00039293,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        *tf_purefock_simulators,
    ),
)
def test_Kerr_gate_leaves_fock_probabilities_invariant(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 5)
        pq.Q(1) | pq.Squeezing(r=0.2, phi=np.pi / 6)

        pq.Q(0) | pq.Kerr(xi=-0.1)
        pq.Q(1) | pq.Kerr(xi=0.2)

    simulator = SimulatorClass(d=2)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [0.97613795, 0.0, 0.0, 0.00484834, 0.0, 0.01901371, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        *tf_purefock_simulators,
    ),
)
def test_Kerr_equivalence(SimulatorClass):
    xi = np.pi / 4

    n = 2

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0]) / np.sqrt(2)
        pq.Q(all) | pq.StateVector([n]) / np.sqrt(2)

        pq.Q(all) | pq.Kerr(xi=xi)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=n + 1))

    state = simulator.execute(program).state

    eixi = np.exp(1j * xi * n * n) / 2
    emixi = np.exp(-1j * xi * n * n) / 2

    assert np.allclose(
        state.density_matrix, [[0.5, 0, emixi], [0, 0, 0], [eixi, 0, 0.5]]
    )


def test_Kerr_equivalence_for_FockSimulator():
    xi = np.pi / 4

    n = 2

    with pq.Program() as program:
        pq.Q(all) | pq.DensityMatrix(ket=(0,), bra=(0,)) / 2
        pq.Q(all) | pq.DensityMatrix(ket=(0,), bra=(2,)) / 2
        pq.Q(all) | pq.DensityMatrix(ket=(2,), bra=(0,)) / 2
        pq.Q(all) | pq.DensityMatrix(ket=(2,), bra=(2,)) / 2

        pq.Q(0) | pq.Kerr(xi=xi)

    simulator = pq.FockSimulator(d=1, config=pq.Config(cutoff=n + 1))

    state = simulator.execute(program).state

    eixi = np.exp(1j * xi * n * n) / 2
    emixi = np.exp(-1j * xi * n * n) / 2

    assert np.allclose(
        state.density_matrix, [[0.5, 0, emixi], [0, 0, 0], [eixi, 0, 0.5]]
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
    ),
)
def test_Attenuator_equivalence_on_one_mode(SimulatorClass):
    config = pq.Config(cutoff=7)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1)

        pq.Q(0) | pq.Phaseshifter(phi=np.pi / 12)

        pq.Q(0) | pq.Attenuator(theta=np.pi / 3)

    simulator = SimulatorClass(d=1, config=config)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [0.9978124, 0.00186894, 0.00031674, 0.00000177, 0.00000015, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
    ),
)
def test_Attenuator_equivalence(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.01)
        pq.Q(1) | pq.Displacement(r=0.02)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(0) | pq.Attenuator(theta=np.pi / 3)

    simulator = SimulatorClass(d=2)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [
            0.99965195,
            0.00005061,
            0.00029737,
            0.0,
            0.00000002,
            0.00000004,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass", (pq.GaussianSimulator, pq.PureFockSimulator, pq.FockSimulator)
)
def test_Attenuator_and_Beamsplitter_with_ancilla_qumode_equivalence(SimulatorClass):
    theta = np.pi / 3

    with pq.Program() as lossy_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(0.03)

        pq.Q(0) | pq.Attenuator(theta=theta)

    with pq.Program() as program_with_ancilla_qumode:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(0.03)

        pq.Q(0, 2) | pq.Beamsplitter(theta=theta, phi=np.pi / 4)

    lossy_simulator = SimulatorClass(d=2)
    ancilla_simulator = SimulatorClass(d=3)

    lossy_state = lossy_simulator.execute(lossy_program).state

    state_with_ancilla_qumode = ancilla_simulator.execute(
        program_with_ancilla_qumode
    ).state
    reduced_state = state_with_ancilla_qumode.reduced(modes=(0, 1))

    assert reduced_state == lossy_state


@pytest.mark.parametrize(
    "SimulatorClass", (pq.GaussianSimulator, pq.PureFockSimulator, pq.FockSimulator)
)
def test_Attenuator_raises_InvalidInstruction_for_multiple_modes(SimulatorClass):
    theta = np.pi / 6
    mean_thermal_excitation = 5

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Attenuator(
            theta=theta, mean_thermal_excitation=mean_thermal_excitation
        )

    simulator = SimulatorClass(d=2)

    with pytest.raises(pq.api.exceptions.InvalidInstruction) as error:
        simulator.execute(program)

    assert "The instruction should be specified for '2' modes:" in error.value.args[0]


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.FockSimulator))
def test_Attenuator_raises_InvalidParam_for_non_zero_mean_thermal_excitation(
    SimulatorClass,
):
    theta = np.pi / 6
    nonzero_mean_thermal_excitation = 5

    with pq.Program() as program:
        pq.Q(0) | pq.Attenuator(
            theta=theta, mean_thermal_excitation=nonzero_mean_thermal_excitation
        )

    simulator = SimulatorClass(d=2)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.execute(program)

    assert (
        "Non-zero mean thermal excitation is not supported in this backend."
        in error.value.args[0]
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_Interferometer_smaller_than_system_size(SimulatorClass):
    config = pq.Config(cutoff=3)

    d = 5

    interferometer_matrix = np.array(
        [
            [
                0.67622072 - 0.02632995j,
                -0.41993552 - 0.43404319j,
                0.21135489 + 0.36417311j,
            ],
            [
                0.48143126 - 0.18351759j,
                -0.05256469 + 0.2714796j,
                -0.7896096 - 0.18600457j,
            ],
            [
                0.47764293 - 0.22007895j,
                0.74644107 + 0.0402763j,
                0.35346034 - 0.19922808j,
            ],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        for i in range(d):
            pq.Q(i) | pq.Squeezing(r=0.01 * i)

        pq.Q(0, 4, 2) | pq.Interferometer(interferometer_matrix)

    simulator = SimulatorClass(d=5, config=config)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [
            0.99850342,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00015807,
            0.0,
            0.0002202,
            0.0,
            0.00011654,
            0.00004992,
            0.0,
            0.0,
            0.0,
            0.00028562,
            0.0,
            0.00016609,
            0.00044906,
            0.0,
            0.00005109,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_Squeezing_with_zero_parameter(SimulatorClass):
    config = pq.Config(cutoff=3)

    d = 2

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.0)
        pq.Q(1) | pq.Squeezing(r=0.1)

    simulator = SimulatorClass(d=d, config=config)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [0.99505769, 0.0, 0.0, 0.0, 0.0, 0.00494231],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_Displacement_with_zero_parameter(SimulatorClass):
    config = pq.Config(cutoff=3)

    d = 2

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.0)
        pq.Q(1) | pq.Displacement(r=0.1)

    simulator = SimulatorClass(d=d, config=config)

    state = simulator.execute(program).state

    assert is_proportional(
        state.fock_probabilities,
        [0.99005, 0.0, 0.0099005, 0.0, 0.0, 0.0000495],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_beamsplitter5050_equivalence(SimulatorClass):
    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 5)
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 7, phi=np.pi / 11)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Beamsplitter5050()

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=5))
    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [
            0.98512079,
            0.0012306,
            0.0086206,
            0.00364572,
            0.00128432,
            0.00001217,
            0.00001391,
            0.00001611,
            0.00001785,
            0.00000122,
            0.00002023,
            0.00001428,
            0.00000213,
            0.00000005,
            0.0,
        ],
    )


@pytest.mark.monkey
@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.SamplingSimulator))
def test_post_select_NS_gate(SimulatorClass):
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = SimulatorClass(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 0.25)

    assert np.isclose(state.norm, 0.25)

    state.normalize()
    state.validate()

    purity = state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_state_vector = np.copy(first_mode_state_vector)
    expected_state_vector[2] *= -1

    assert np.allclose(
        state.density_matrix[:d, :d],
        np.outer(expected_state_vector, expected_state_vector),
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator, pq.SamplingSimulator),
)
def test_post_select_random_unitary(SimulatorClass):
    d = 3

    interferometer_matrix = unitary_group.rvs(d)

    postselect_modes = (1, 2)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
        pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
        pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=postselect_modes, photon_counts=(1, 0)
        )

    simulator = SimulatorClass(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    state.normalize()
    state.validate()

    assert state.d == d - len(postselect_modes)


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator, pq.SamplingSimulator),
)
def test_post_select_conditional_sign_flip_gate_with_1_over_16_success_rate(
    SimulatorClass,
):
    ancilla_modes = (4, 5, 6, 7)

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as nonlinear_shift:
        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

    with pq.Program() as conditional_sign_flip:
        pq.Q(0, 2) | pq.Beamsplitter(theta=np.pi / 4)

        pq.Q(0, ancilla_modes[0], ancilla_modes[1]) | nonlinear_shift
        pq.Q(2, ancilla_modes[2], ancilla_modes[3]) | nonlinear_shift

        pq.Q(0, 2) | pq.Beamsplitter(theta=-np.pi / 4)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    states = [state_00, state_01, state_10, state_11]
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    ancilla_state = [1, 0]

    with pq.Program() as program:
        for input_state, coeff in zip(states, coefficients):
            pq.Q(all) | pq.StateVector(input_state + ancilla_state * 2) * coeff

        pq.Q(all) | conditional_sign_flip

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=ancilla_modes, photon_counts=ancilla_state * 2
        )

    simulator = SimulatorClass(d=8, config=pq.Config(cutoff=5))

    final_state = simulator.execute(program).state

    expected_success_rate = 1 / 16
    actual_success_rate = final_state.norm

    assert np.isclose(expected_success_rate, actual_success_rate)

    final_state.normalize()

    purity = final_state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    expected_coefficients[3] *= -1

    with pq.Program() as expectation_program:
        for input_state, coeff in zip(states, expected_coefficients):
            pq.Q(all) | pq.StateVector(input_state) * coeff

    expectation_simulator = SimulatorClass(d=4, config=pq.Config(cutoff=5))
    expected_state = expectation_simulator.execute(expectation_program).state

    assert np.allclose(expected_state.density_matrix, final_state.density_matrix)


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator, pq.SamplingSimulator),
)
def test_ImperfectPostSelectPhotons(SimulatorClass):
    d = 5
    cutoff = 4

    detector_efficiency_matrix = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 0.7],
        ]
    )

    coeffs = np.sqrt([0.1, 0.3, 0.4, 0.05, 0.1, 0.05])

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0, 0, 2, 1]) * coeffs[0]
        pq.Q() | pq.StateVector([0, 0, 2, 0, 1]) * coeffs[1]
        pq.Q() | pq.StateVector([0, 1, 0, 1, 1]) * coeffs[2]
        pq.Q() | pq.StateVector([1, 1, 0, 1, 0]) * coeffs[3]
        pq.Q() | pq.StateVector([3, 0, 0, 0, 0]) * coeffs[4]

        pq.Q() | pq.ImperfectPostSelectPhotons(
            postselect_modes=(2, 4),
            photon_counts=(0, 1),
            detector_efficiency_matrix=detector_efficiency_matrix,
        )

    simulator = SimulatorClass(d=d, config=pq.Config(cutoff=cutoff))

    state = simulator.execute(program).state

    with pq.Program() as expected_program:
        pq.Q() | pq.DensityMatrix((0, 1, 1), (0, 1, 1)) * 0.32
        pq.Q() | pq.DensityMatrix((0, 1, 1), (0, 0, 2)) * 0.16
        pq.Q() | pq.DensityMatrix((0, 0, 2), (0, 1, 1)) * 0.16
        pq.Q() | pq.DensityMatrix((0, 0, 2), (0, 0, 2)) * 0.08

    expected_simulator = pq.FockSimulator(d=3, config=pq.Config(cutoff=4))

    expected_state = expected_simulator.execute(expected_program).state

    assert state == expected_state


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator, pq.SamplingSimulator),
)
def test_NS_gate_with_ImperfectPostSelectPhotons_trivial_case(SimulatorClass):
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    trivial_detector_efficiency_matrix = np.identity(d)

    with pq.Program() as preparation:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

    with pq.Program() as imperfect_photon_detection_program:
        pq.Q(all) | preparation

        pq.Q(all) | pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=trivial_detector_efficiency_matrix,
        )

    with pq.Program() as perfect_photon_detection_program:
        pq.Q(all) | preparation

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = SimulatorClass(d=d, config=pq.Config(cutoff=4))

    imperfect_state = simulator.execute(imperfect_photon_detection_program).state
    perfect_state = simulator.execute(perfect_photon_detection_program).state

    imperfect_success_rate = imperfect_state.norm
    perfect_success_rate = perfect_state.norm
    assert np.isclose(imperfect_success_rate, perfect_success_rate)
    assert np.isclose(perfect_success_rate, 1 / 4)

    imperfect_state.normalize()
    perfect_state.normalize()

    imperfect_state.validate()
    perfect_state.validate()

    assert np.isclose(imperfect_state.get_purity(), 1.0)

    assert np.allclose(imperfect_state.density_matrix, perfect_state.density_matrix)


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator, pq.SamplingSimulator),
)
def test_NS_gate_with_ImperfectPostSelectPhotons(SimulatorClass):
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    detector_efficiency_matrix = np.array(
        [[1.0, 0.1, 0.2], [0.0, 0.9, 0.2], [0.0, 0.0, 0.6]]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

        pq.Q(all) | pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=detector_efficiency_matrix,
        )

    simulator = SimulatorClass(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    success_rate = state.norm

    assert np.isclose(success_rate, 0.25773863607376246)

    state.normalize()
    state.validate()

    assert np.isclose(state.get_purity(), 0.8645456946151449)

    assert np.allclose(
        state.density_matrix,
        [
            [0.25690057, 0.25784735, -0.27605969, 0.0],
            [0.25784735, 0.30661073, -0.33810269, 0.0],
            [-0.27605969, -0.33810269, 0.43648869, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_mean_photon_number_equivalence(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 5)
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 5)

        pq.Q(0) | pq.Phaseshifter(np.pi / 9)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 7, phi=np.pi / 11)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state

    assert np.allclose(state.mean_photon_number(), 0.02003337780953792)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_variance_photon_number_equivalence_1_mode(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 5)

        pq.Q(0) | pq.Phaseshifter(np.pi / 9)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state

    assert np.allclose(state.variance_photon_number(), 0.01)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.GaussianSimulator,
        *tf_purefock_simulators,
    ),
)
def test_variance_photon_number_equivalence(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.1, phi=np.pi / 5)
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 5)

        pq.Q(0) | pq.Phaseshifter(np.pi / 9)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 7, phi=np.pi / 11)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state

    assert np.allclose(state.variance_photon_number(), 0.030268090551006963)
