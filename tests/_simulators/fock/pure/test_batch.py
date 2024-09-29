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


def test_BatchPureFockState_equals_to_itself():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state = simulator.execute(batch_program).state

    assert batch_state == batch_state


def test_BatchPureFockState_without_normalization():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state = simulator.execute(batch_program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        batch_state.validate()

    assert error.value.args[0] == (
        "The sum of probabilities is not close to 1.0 for at least one state in the "
        "batch."
    )


def test_BatchPureFockState_without_normalization_but_validate_False():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5, validate=False))

    batch_state = simulator.execute(batch_program).state

    batch_state.validate()


def test_invalid_BatchPureFockState_normalize():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as invalid_second_preparation:
        pass

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, invalid_second_preparation])

        pq.Q() | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state = simulator.execute(batch_program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        batch_state.normalize()

    assert "The norm of a state in the batch is 0." == error.value.args[0]


def test_batch_Beamsplitter_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Squeezing_and_Displacement_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Squeezing(0.1)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Kerr_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Phaseshifter_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0) | pq.Phaseshifter(0.1)
        pq.Q(1) | pq.Phaseshifter(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_multiple_gates_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Phaseshifter(0.1)
        pq.Q(1) | pq.Phaseshifter(0.2)

        pq.Q(0) | pq.Squeezing(0.1)
        pq.Q(0) | pq.Squeezing(0.2)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 7)

        pq.Q(0) | pq.Phaseshifter(0.5)
        pq.Q(1) | pq.Phaseshifter(0.2)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Displacement(1.0, phi=np.pi / 2)

        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_multiple_gates_nonzero_elements():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Phaseshifter(0.1)
        pq.Q(1) | pq.Phaseshifter(0.2)

        pq.Q(0) | pq.Squeezing(0.1)
        pq.Q(0) | pq.Squeezing(0.2)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 7)

        pq.Q(0) | pq.Phaseshifter(0.5)
        pq.Q(1) | pq.Phaseshifter(0.2)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Displacement(1.0, phi=np.pi / 2)

        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_as_string = str(simulator.execute(batch_program).state)

    first_state_as_string = str(simulator.execute(first_program).state)
    second_state_as_string = str(simulator.execute(second_program).state)

    assert batch_state_as_string == f"{first_state_as_string}\n{second_state_as_string}"


def test_batch_mean_position():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Squeezing(0.1)

        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    first_mean_position = simulator.execute(first_program).state.mean_position(0)
    second_mean_position = simulator.execute(second_program).state.mean_position(0)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)


def test_Batch_with_OneByOne():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common1:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Squeezing(0.1)

    with pq.Program() as common2:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

        pq.Q(0) | pq.Displacement(-1.0, phi=np.pi / 3)
        pq.Q(0) | pq.Squeezing(0.2, phi=np.pi / 5)

    with pq.Program() as first_intermediate:
        pq.Q(0) | pq.Kerr(0.1)

    with pq.Program() as second_intermediate:
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

        pq.Q() | common1

        pq.Q() | pq.BatchApply([first_intermediate, second_intermediate])

        pq.Q() | common2

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common1

        pq.Q() | first_intermediate

        pq.Q() | common2

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common1

        pq.Q() | second_intermediate

        pq.Q() | common2

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state.state_vector

    first_state_vector = simulator.execute(first_program).state.state_vector
    second_state_vector = simulator.execute(second_program).state.state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)
