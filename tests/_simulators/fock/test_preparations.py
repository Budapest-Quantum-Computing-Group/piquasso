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

import numpy as np

import piquasso as pq


def test_from_fock_state_preserves_fock_probabilities():
    with pq.Program() as pure_state_preparation_program:
        pq.Q(1) | pq.StateVector([1])

    pure_simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=4))
    mixed_simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=4))
    pure_state_preparation_state = pure_simulator.execute(
        pure_state_preparation_program
    ).state

    beamsplitter = pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 3)

    with pq.Program() as pure_state_program:
        pq.Q(0, 1) | beamsplitter

    with pq.Program() as mixed_state_program:
        pq.Q(0, 1) | beamsplitter

    pure_state = pure_simulator.execute(
        pure_state_program,
        initial_state=pure_state_preparation_state,
    ).state

    mixed_state = mixed_simulator.execute(
        mixed_state_program,
        initial_state=pq.FockState.from_fock_state(pure_state_preparation_state),
    ).state

    pure_state_fock_probabilities = pure_state.fock_probabilities
    mixed_state_fock_probabilities = mixed_state.fock_probabilities

    assert np.allclose(mixed_state_fock_probabilities, pure_state_fock_probabilities)
