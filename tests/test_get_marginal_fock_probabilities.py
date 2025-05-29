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


def test_pure_fock_state_get_marginal_fock_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        
        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.2)
        
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))
    result = simulator.execute(program)
    
    # Get marginal probabilities for mode 0
    marginal_probs = result.state.get_marginal_fock_probabilities(modes=(0,))
    
    # Verify shape and properties
    assert len(marginal_probs) == 3  # Should match cutoff
    assert np.isclose(np.sum(marginal_probs), 1.0)  # Should be normalized


def test_general_fock_state_get_marginal_fock_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        
        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.2)
        
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)
        
        # Add some noise to make it a mixed state
        pq.Q(0) | pq.Attenuator(theta=0.1)

    simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=3))
    result = simulator.execute(program)
    
    # Get marginal probabilities for mode 1
    marginal_probs = result.state.get_marginal_fock_probabilities(modes=(1,))
    
    # Verify shape and properties
    assert len(marginal_probs) == 3  # Should match cutoff
    assert np.isclose(np.sum(marginal_probs), 1.0)  # Should be normalized


def test_multi_mode_marginal_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        
        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.2)
        pq.Q(2) | pq.Displacement(r=0.5)
        
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=3))
    result = simulator.execute(program)
    
    # Get marginal probabilities for modes 0 and 2
    marginal_probs = result.state.get_marginal_fock_probabilities(modes=(0, 2))
    
    # Verify shape and properties
    assert marginal_probs.shape == (3, 3)  # Should be 2D for two modes
    assert np.isclose(np.sum(marginal_probs), 1.0)  # Should be normalized


def test_full_system_marginal_returns_same_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        
        pq.Q(0) | pq.Squeezing(r=0.1)
        pq.Q(1) | pq.Squeezing(r=0.2)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=3))
    result = simulator.execute(program)
    
    # Get marginal probabilities for all modes
    marginal_probs = result.state.get_marginal_fock_probabilities(modes=(0, 1))
    full_probs = result.state.fock_probabilities
    
    # Should be the same as the full probabilities
    assert np.allclose(marginal_probs.flatten(), full_probs)