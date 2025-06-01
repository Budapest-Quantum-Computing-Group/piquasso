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


def is_proportional(first, second, atol=1e-3):
    first = np.array(first)
    second = np.array(second)

    index = np.argmax(first)

    # proportion = first[index] / second[index]

    return np.allclose(first, second, atol=atol)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_multiple_modes(SimulatorClass):
    """Test marginal probabilities for multiple modes from a multi-mode state."""

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.15)
        pq.Q(2) | pq.Squeezing(r=0.05)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 8)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    # Get marginal probabilities for modes (0, 1)
    marginal_probs_01 = state.get_marginal_fock_probabilities(modes=(0, 1))

    # Get marginal probabilities for modes (1, 2)
    marginal_probs_12 = state.get_marginal_fock_probabilities(modes=(1, 2))

    # Check that probabilities sum to 1
    assert np.isclose(np.sum(marginal_probs_01), 1.0, atol=1e-02)
    assert np.isclose(np.sum(marginal_probs_12), 1.0, atol=1e-02)

    # Check that probabilities are non-negative
    assert np.all(marginal_probs_01 >= 0)
    assert np.all(marginal_probs_12 >= 0)

    # For 2-mode system with cutoff=4, we have 10 Fock basis states:
    # |0,0⟩, |0,1⟩, |0,2⟩, |0,3⟩, |1,0⟩, |1,1⟩, |1,2⟩, |2,0⟩, |2,1⟩, |3,0⟩
    # (states where n1 + n2 < cutoff)

    # Test with explicit expected values for modes (0, 1)

    assert is_proportional(
        marginal_probs_01,
        [
            0.984520983,
            8.71509e-05,
            0.000678455,
            0.007588746,
            0.000507953,
            0.006317764,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    assert is_proportional(
        marginal_probs_12,
        [
            0.990290870,
            0.000507953,
            8.71509e-05,
            0.006317764,
            0.000678455,
            0.001818859,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


@pytest.mark.parametrize(
    "SimulatorClass",
    (pq.PureFockSimulator,),
)
def test_get_marginal_fock_probabilities_single_photon_state(SimulatorClass):
    """Test marginal probabilities for single photon state."""

    theta = np.pi / 3

    simulator = SimulatorClass(
        d=2,
        config=pq.Config(cutoff=3),
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector((2, 0))

        pq.Q(all) | pq.Beamsplitter(theta=theta)

    state = simulator.execute(program).state

    assert np.allclose(
        state.get_marginal_fock_probabilities(modes=(0,)),
        [
            np.sin(theta) ** 4,
            2 * (np.cos(theta) * np.sin(theta)) ** 2,
            np.cos(theta) ** 4,
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
def test_get_marginal_fock_probabilities_all_modes(SimulatorClass):
    """Test that marginal probabilities for all modes equals full state probabilities."""

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.05)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    # Get marginal probabilities for all modes
    marginal_probs_all = state.get_marginal_fock_probabilities(modes=(0, 1))

    # Get full state probabilities
    full_probs = state.fock_probabilities

    # They should be equal (within numerical precision)
    assert np.allclose(marginal_probs_all, full_probs)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_consistency_with_reduced_state(SimulatorClass):
    """Test that marginal probabilities are consistent with reduced state probabilities."""

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.2)
        pq.Q(1) | pq.Squeezing(r=0.1)
        pq.Q(2) | pq.Squeezing(r=0.15)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=5))
    state = simulator.execute(program).state

    # Get marginal probabilities using the method
    marginal_probs = state.get_marginal_fock_probabilities(modes=(0, 2))

    # Get probabilities from reduced state
    reduced_state = state.reduced(modes=(0, 2))
    reduced_probs = reduced_state.fock_probabilities

    # They should be equal
    assert np.allclose(marginal_probs, reduced_probs)
