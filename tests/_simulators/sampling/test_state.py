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

import numpy as np
import pytest

import piquasso as pq


def test_state_vector(generate_unitary_matrix):
    input_state = np.array([2, 1, 3, 1, 1], dtype=int)

    unitary = generate_unitary_matrix(5)

    program = pq.Program(
        instructions=[
            pq.StateVector(input_state),
            pq.Interferometer(unitary),
        ]
    )

    simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=9))

    result = simulator.execute(program)

    state = result.state

    state.validate()

    state_vector = state.state_vector

    assert np.isclose(np.sum(np.abs(state_vector) ** 2), 1.0)


def test_validate_not_normalized():
    input_state = [1, 1]

    program = pq.Program(
        instructions=[
            pq.StateVector(input_state),
            pq.Beamsplitter5050().on_modes(0, 1),
            pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
        ]
    )

    simulator = pq.SamplingSimulator(config=pq.Config(cutoff=3))

    result = simulator.execute(program)

    with pytest.raises(
        pq.api.exceptions.InvalidState, match="The state is not normalized."
    ):
        result.state.validate()
