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

import random
import pytest
import numpy as np
import piquasso as pq

from scipy.linalg import polar, sinhm, coshm

from functools import partial


CUTOFF = 4


def is_proportional(first, second):
    first = np.array(first)
    second = np.array(second)

    index = np.argmax(first)

    proportion = first[index] / second[index]

    return np.allclose(first, proportion * second)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_squeezed_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_displaced_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_displaced_state_with_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_squeezed_state_with_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_two_mode_squeezing(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_two_mode_squeezing_and_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0, 1) | pq.Squeezing2(r=0.1, phi=0.6)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_general_gaussian_transform(StateClass):
    from scipy.linalg import polar, sinhm, coshm, expm
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
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0, 1) | pq.GaussianTransform(P=passive, A=active)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)
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
def test_monkey_get_fock_probabilities_with_general_gaussian_transform(
    generate_unitary_matrix
):
    d = 3

    # TODO: General symmetric rotation squeezing matrices don't do the trick for
    # some reason: there might be an issue with the GaussianState, because we get
    # negative probabilities in that case.
    squeezing_param = random.uniform(0, 1) + 1j * random.uniform(0, 1)
    some_unitary = generate_unitary_matrix(d)
    squeezing_matrix = (
        some_unitary
        @ np.diag([squeezing_param] * d)
        @ some_unitary.conjugate().transpose()
    )

    U, r = polar(squeezing_matrix)

    global_phase = generate_unitary_matrix(d)
    passive = global_phase @ coshm(r)
    active = global_phase @ U @ sinhm(r)

    with pq.Program() as fock_program:
        pq.Q() | pq.FockState(d=d, cutoff=CUTOFF) | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(P=passive, A=active)

    fock_program.execute()

    fock_representation_probabilities = (
        fock_program.state.get_fock_probabilities(cutoff=CUTOFF)
    )

    with pq.Program() as gaussian_program:
        pq.Q() | pq.GaussianState(d=d) | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(P=passive, A=active)

    gaussian_program.execute()

    gaussian_representation_probabilities = (
        gaussian_program.state.get_fock_probabilities(cutoff=CUTOFF)
    )

    normalization = 1 / sum(gaussian_representation_probabilities)

    assert np.allclose(
        fock_representation_probabilities,
        normalization * gaussian_representation_probabilities,
    )
