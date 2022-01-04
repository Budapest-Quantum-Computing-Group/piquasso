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


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.FockSimulator))
def test_squeezing_probabilities(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.5, phi=np.pi / 3)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=3))
    state = simulator.execute(program).state

    state.validate()

    assert np.isclose(
        sum(state.fock_probabilities), 1.0
    ), "The state should be renormalized for probability conservation."

    assert np.allclose(state.fock_probabilities, [0.90352508, 0, 0, 0, 0, 0.09647492])


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.FockSimulator))
def test_displacement_probabilities(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5, phi=np.pi / 3)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    state.validate()

    assert np.isclose(
        sum(state.fock_probabilities), 1.0
    ), "The state should be renormalized for probability conservation."

    assert np.allclose(
        state.fock_probabilities, [0.7804878, 0, 0.19512195, 0, 0, 0.02439024]
    )


@pytest.mark.parametrize("SimulatorClass", (pq.PureFockSimulator, pq.FockSimulator))
def test_displacement_get_fock_prob(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.2, phi=np.pi / 3)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=8))

    state = simulator.execute(program).state

    state.validate()
    assert np.isclose(
        state.get_particle_detection_probability(occupation_number=(0, 1)), 0.0
    )


def test_PureFockState_displacement_on_single_mode():

    alpha = 2.0
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=alpha)

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=20))

    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 20
    np.isclose(nonzero_elements[0][0], np.exp(-0.5 * np.abs(alpha ** 2)))


def test_FockState_displacement_on_single_mode():

    alpha = 2.0
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=alpha)

    simulator = pq.FockSimulator(d=1, config=pq.Config(cutoff=20))

    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 400.0
    np.isclose(nonzero_elements[0][0], np.exp(-0.5 * np.abs(alpha ** 2)))


def test_PureFockState_squeezing_on_single_mode():
    r = 0.5
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=r, phi=phi)

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 2

    normalization = 1 / np.sqrt((1 / np.cosh(r)) * (1 + np.tanh(r) ** 2 / 2))

    assert nonzero_elements[0][1] == (0,)
    assert np.isclose(nonzero_elements[0][0], normalization * 1 / np.sqrt(np.cosh(r)))

    assert nonzero_elements[1][1] == (2,)
    assert np.isclose(
        nonzero_elements[1][0],
        normalization
        * (-np.exp(1j * phi) * np.tanh(r) / np.sqrt(2))
        / np.sqrt(np.cosh(r)),
    )


def test_PureFockState_squeezing():
    r = 0.5
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=r, phi=phi)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 2

    normalization = 1 / np.sqrt((1 / np.cosh(r)) * (1 + np.tanh(r) ** 2 / 2))

    assert nonzero_elements[0][1] == (0, 0)
    assert np.isclose(nonzero_elements[0][0], normalization * 1 / np.sqrt(np.cosh(r)))

    assert nonzero_elements[1][1] == (2, 0)
    assert np.isclose(
        nonzero_elements[1][0],
        normalization
        * (-np.exp(1j * phi) * np.tanh(r) / np.sqrt(2))
        / np.sqrt(np.cosh(r)),
    )


def test_PureFockState_displacement():
    alpha = 0.5 * np.exp(1j * np.pi / 3)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=alpha)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 3

    normalization = 1 / np.sqrt(
        np.exp(-np.abs(alpha) ** 2)
        * (1 + np.abs(alpha) ** 2 + np.abs(alpha) ** 4 / np.sqrt(4))
    )

    assert nonzero_elements[0][1] == (0, 0)
    assert np.isclose(
        nonzero_elements[0][0], normalization * np.exp(-np.abs(alpha) ** 2 / 2)
    )

    assert nonzero_elements[1][1] == (1, 0)
    assert np.isclose(
        nonzero_elements[1][0], normalization * np.exp(-np.abs(alpha) ** 2 / 2) * alpha
    )

    assert nonzero_elements[2][1] == (2, 0)
    assert np.isclose(
        nonzero_elements[2][0],
        normalization * np.exp(-np.abs(alpha) ** 2 / 2) * (alpha ** 2) / np.sqrt(2),
    )


def test_FockState_squeezing():
    r = 0.5
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=r, phi=phi)

    simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=3))
    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    vacuum_probability = 1 / np.cosh(r)

    two_particle_probability = (1 / np.cosh(r)) * np.tanh(r) ** 2 / 2

    normalization = 1 / (vacuum_probability + two_particle_probability)

    assert len(nonzero_elements) == 4

    assert nonzero_elements[0][1] == ((0, 0), (0, 0))
    assert np.isclose(nonzero_elements[0][0], normalization * vacuum_probability)

    assert nonzero_elements[1][1] == ((0, 0), (2, 0))
    assert np.isclose(
        nonzero_elements[1][0],
        normalization * (-np.exp(-1j * phi) * np.tanh(r) / np.sqrt(2)) / np.cosh(r),
    )

    assert nonzero_elements[2][1] == ((2, 0), (0, 0))
    assert np.isclose(
        nonzero_elements[2][0],
        normalization * (-np.exp(1j * phi) * np.tanh(r) / np.sqrt(2)) / np.cosh(r),
    )

    assert nonzero_elements[3][1] == ((2, 0), (2, 0))
    assert np.isclose(nonzero_elements[3][0], normalization * two_particle_probability)


def test_FockState_displacement():
    alpha = 0.5 * np.exp(1j * np.pi / 3)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=alpha)

    simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=2))
    state = simulator.execute(program).state

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(state.nonzero_elements)

    assert len(nonzero_elements) == 4

    vacuum_probability = np.exp(-np.abs(alpha) ** 2)

    one_particle_probability = np.exp(-np.abs(alpha) ** 2) * np.abs(alpha) ** 2

    normalization = 1 / (vacuum_probability + one_particle_probability)

    assert nonzero_elements[0][1] == ((0, 0), (0, 0))
    assert np.isclose(nonzero_elements[0][0], normalization * vacuum_probability)

    assert nonzero_elements[1][1] == ((0, 0), (1, 0))
    assert np.isclose(
        nonzero_elements[1][0],
        normalization * np.exp(-np.abs(alpha) ** 2) * alpha.conj(),
    )

    assert nonzero_elements[2][1] == ((1, 0), (0, 0))
    assert np.isclose(
        nonzero_elements[2][0], normalization * np.exp(-np.abs(alpha) ** 2) * alpha
    )

    assert nonzero_elements[3][1] == ((1, 0), (1, 0))
    assert np.isclose(
        nonzero_elements[3][0],
        normalization * one_particle_probability,
    )
