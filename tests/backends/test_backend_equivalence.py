#
# Copyright 2021 Budapest Quantum Computing Group
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
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_should_be_numpy_array_of_floats(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.dtype == np.float64


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_squeezed_state(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    state = StateClass(d=3)
    state._config.cutoff = 4
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99502075,
            0., 0., 0.,
            0., 0., 0., 0., 0., 0.00494212,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    )


def test_density_matrix_with_squeezed_state():
    d = 2

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

    gaussian_state = pq.GaussianState(d=d)
    gaussian_state.apply(gaussian_program)

    gaussian_density_matrix = gaussian_state.density_matrix

    normalization = 1 / sum(np.diag(gaussian_density_matrix))

    with pq.Program() as fock_program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

    fock_state = pq.FockState(d=d)
    fock_state.apply(fock_program)

    fock_density_matrix = fock_state.density_matrix

    assert np.allclose(normalization * gaussian_density_matrix, fock_density_matrix)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_displaced_state(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0., 0., 0.03368973,
            0., 0., 0., 0., 0., 0.08422434,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1403739,
        ]
    )


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_displaced_state_with_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0., 0.0252673, 0.00842243,
            0., 0., 0.04737619, 0., 0.03158413, 0.00526402,
            0., 0., 0., 0.05922024, 0., 0., 0.05922024, 0., 0.01974008, 0.00219334
        ]
    )


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_squeezed_state_with_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99502075,
            0., 0., 0.,
            0., 0., 0.00277994, 0., 0.0018533, 0.00030888,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    )


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_two_mode_squeezing(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99006629,
            0., 0., 0.,
            0., 0., 0., 0., 0.00983503, 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    )


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_two_mode_squeezing_and_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99006629,
            0., 0., 0.,
            0., 0., 0.00368814, 0., 0.00245876, 0.00368814,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    )


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_quadratic_phase(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.QuadraticPhase(s=0.4)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.98058068,
        0., 0., 0.,
        0., 0., 0., 0., 0., 0.01885732,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_position_displacement(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.PositionDisplacement(x=0.2)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.96078944,
        0., 0., 0.03843158,
        0., 0., 0., 0., 0., 0.00076863,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00001025,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_momentum_displacement(StateClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.MomentumDisplacement(p=0.2)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.96078944,
        0., 0., 0.03843158,
        0., 0., 0., 0., 0., 0.00076863,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00001025,
    ]

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(probabilities, expected_probabilities)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_position_displacement_is_HBAR_independent(
    StateClass
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.PositionDisplacement(x=0.4)

    state1 = StateClass(d=3)
    state2 = StateClass(d=3)

    state1._config.hbar = 2
    state2._config.hbar = 42

    state1.apply(program)
    state2.apply(program)

    probabilities1 = state1.fock_probabilities
    probabilities2 = state1.fock_probabilities

    assert np.allclose(probabilities1, probabilities2)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_momentum_displacement_is_HBAR_independent(
    StateClass
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.MomentumDisplacement(p=0.4)

    state1 = StateClass(d=3)
    state2 = StateClass(d=3)

    state1._config.hbar = 2
    state2._config.hbar = 42

    state1.apply(program)
    state2.apply(program)

    probabilities1 = state1.fock_probabilities
    probabilities2 = state1.fock_probabilities

    assert np.allclose(probabilities1, probabilities2)


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        pq.PureFockState,
        pq.FockState,
    )
)
def test_fock_probabilities_with_general_gaussian_transform(StateClass):
    squeezing_matrix = np.array(
        [
            [0.1, 0.2 + 0.3j],
            [0.2 + 0.3j, 0.1],
        ],
        dtype=complex
    )

    rotation_matrix = np.array(
        [
            [1, 3 - 2j],
            [3 + 2j, 1],
        ],
        dtype=complex
    )

    U, r = polar(squeezing_matrix)

    passive = expm(-1j * rotation_matrix) @ coshm(r)
    active = expm(-1j * rotation_matrix) @ U @ sinhm(r)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.GaussianTransform(passive=passive, active=active)

    state = StateClass(d=3)
    state.apply(program)

    probabilities = state.fock_probabilities
    expected_probabilities = [
        0.864652,
        0., 0., 0.,
        0., 0., 0.05073686, 0., 0.02118922, 0.0379305,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
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

    fock_state = pq.FockState(d=d)
    fock_state.apply(fock_program)

    fock_representation_probabilities = (
        fock_state.fock_probabilities
    )

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    gaussian_state = pq.GaussianState(d=d)
    gaussian_state.apply(gaussian_program)

    gaussian_representation_probabilities = (
        gaussian_state.fock_probabilities
    )

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

    fock_state = pq.FockState(d=d)
    fock_state.apply(fock_program)

    fock_representation_probabilities = (
        fock_state.fock_probabilities
    )

    with pq.Program() as gaussian_program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    gaussian_state = pq.GaussianState(d=d)
    gaussian_state.apply(gaussian_program)

    gaussian_representation_probabilities = (
        gaussian_state.fock_probabilities
    )

    normalization = 1 / sum(gaussian_representation_probabilities)

    assert np.allclose(
        fock_representation_probabilities,
        normalization * gaussian_representation_probabilities,
    )


def test_sampling_backend_equivalence_for_two_mode_beamsplitter():
    initial_occupation_numbers = (1, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(*initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

    fock_state = pq.PureFockState(d=d)
    fock_state._config.cutoff = sum(initial_occupation_numbers) + 1
    fock_state.apply(program)

    sampling_state = pq.SamplingState(d=d)
    sampling_state.apply(program)

    assert np.allclose(
        fock_state.fock_probabilities,
        sampling_state.fock_probabilities,
    )


def test_sampling_backend_equivalence_complex_scenario():
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(*initial_occupation_numbers)

        pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter(np.pi / 4)

    fock_state = pq.PureFockState(d=d)
    fock_state._config.cutoff = sum(initial_occupation_numbers) + 1
    fock_state.apply(program)
    fock_state.validate()

    sampling_state = pq.SamplingState(d=d)
    sampling_state.apply(program)
    sampling_state.validate()

    assert np.allclose(
        fock_state.fock_probabilities,
        sampling_state.fock_probabilities
    )


@pytest.mark.monkey
def test_sampling_backend_equivalence_with_random_interferometer(
    generate_unitary_matrix
):
    initial_occupation_numbers = (1, 1, 0, 1)
    d = len(initial_occupation_numbers)

    interferometer_matrix = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(*initial_occupation_numbers)

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

    fock_state = pq.PureFockState(d=d)
    fock_state._config.cutoff = sum(initial_occupation_numbers) + 1
    fock_state.apply(program)
    fock_state.validate()

    sampling_state = pq.SamplingState(d=d)
    sampling_state.apply(program)
    sampling_state.validate()

    assert np.allclose(
        fock_state.fock_probabilities,
        sampling_state.fock_probabilities,
    )
