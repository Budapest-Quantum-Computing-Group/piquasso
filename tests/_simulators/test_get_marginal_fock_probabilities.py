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


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_single_mode(SimulatorClass):
    """Test marginal probabilities for a single mode from a multi-mode state."""
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.2)
        pq.Q(1) | pq.Squeezing(r=0.1)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=5))
    state = simulator.execute(program).state

    # Get marginal probabilities for mode 0
    marginal_probs_mode_0 = state.get_marginal_fock_probabilities(modes=(0,))
    # Get marginal probabilities for mode 1
    marginal_probs_mode_1 = state.get_marginal_fock_probabilities(modes=(1,))

    # Check that probabilities sum to 1
    assert np.isclose(np.sum(marginal_probs_mode_0), 1.0,atol=1e-02)
    assert np.isclose(np.sum(marginal_probs_mode_1), 1.0,atol=1e-02)
    
    # Check that probabilities are non-negative
    assert np.all(marginal_probs_mode_0 >= 0)
    assert np.all(marginal_probs_mode_1 >= 0)


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
    assert np.isclose(np.sum(marginal_probs_01), 1.0,atol=1e-02)
    assert np.isclose(np.sum(marginal_probs_12), 1.0,atol=1e-02)
    
    # Check that probabilities are non-negative
    assert np.all(marginal_probs_01 >= 0)
    assert np.all(marginal_probs_12 >= 0)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_vacuum_state(SimulatorClass):
    """Test marginal probabilities for vacuum state."""
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=3))
    state = simulator.execute(program).state

    # Get marginal probabilities for mode 0
    marginal_probs = state.get_marginal_fock_probabilities(modes=(0,))

    # For vacuum state, only |0⟩ should have probability 1
    expected_probs = np.zeros(3)
    expected_probs[0] = 1.0

    assert np.allclose(marginal_probs, expected_probs)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_single_photon_state(SimulatorClass):
    """Test marginal probabilities for single photon state."""
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(1) | pq.Create()

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=3))
    state = simulator.execute(program).state

    # Get marginal probabilities for mode 1
    marginal_probs_mode_1 = state.get_marginal_fock_probabilities(modes=(1,))
    
    # Get marginal probabilities for mode 0 (should be vacuum)
    marginal_probs_mode_0 = state.get_marginal_fock_probabilities(modes=(0,))

    # Mode 1 should have probability 1 for |1⟩ state
    expected_probs_mode_1 = np.zeros(3)
    expected_probs_mode_1[1] = 1.0
    
    # Mode 0 should have probability 1 for |0⟩ state
    expected_probs_mode_0 = np.zeros(3)
    expected_probs_mode_0[0] = 1.0

    assert np.allclose(marginal_probs_mode_1, expected_probs_mode_1)
    assert np.allclose(marginal_probs_mode_0, expected_probs_mode_0)


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


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_with_displacement(SimulatorClass):
    """Test marginal probabilities with displacement operations."""
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r = 0, phi=0.5)
        pq.Q(1) | pq.Displacement(r = 0, phi=0.3j)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=6))
    state = simulator.execute(program).state

    # Get marginal probabilities for each mode
    marginal_probs_0 = state.get_marginal_fock_probabilities(modes=(0,))
    marginal_probs_1 = state.get_marginal_fock_probabilities(modes=(1,))

    # Check basic properties
    assert np.isclose(np.sum(marginal_probs_0), 1.0,atol=1e-02)
    assert np.isclose(np.sum(marginal_probs_1), 1.0,atol=1e-02)
    assert np.all(marginal_probs_0 >= 0)
    assert np.all(marginal_probs_1 >= 0)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.GaussianSimulator,
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_get_marginal_fock_probabilities_mode_ordering(SimulatorClass):
    """Test that mode ordering doesn't affect the result structure."""
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.2)
        pq.Q(2) | pq.Squeezing(r=0.05)

    simulator = SimulatorClass(d=3, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    # Get marginal probabilities for modes in different orders
    marginal_probs_01 = state.get_marginal_fock_probabilities(modes=(0, 1))
    marginal_probs_10 = state.get_marginal_fock_probabilities(modes=(1, 0))

    # Both should sum to 1 and be non-negative
    assert np.isclose(np.sum(marginal_probs_01), 1.0,atol=1e-02)
    assert np.isclose(np.sum(marginal_probs_10), 1.0,atol=1e-02)
    assert np.all(marginal_probs_01 >= 0)
    assert np.all(marginal_probs_10 >= 0)