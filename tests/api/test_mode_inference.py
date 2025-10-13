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

"""Tests for mode inference feature """

import pytest
import piquasso as pq


def test_program_get_number_of_modes_with_two_modes():
    """Test that get_number_of_modes correctly infers d=2."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))
        pq.Q(0, 1) | pq.Beamsplitter5050()

    assert program.get_number_of_modes() == 2


def test_program_get_number_of_modes_with_single_mode():
    """Test that get_number_of_modes correctly infers d=1."""
    with pq.Program() as program:
        pq.Q(0) | pq.StateVector((2,))

    assert program.get_number_of_modes() == 1


def test_program_get_number_of_modes_with_sparse_modes():
    """Test that get_number_of_modes infers based on max mode index."""
    with pq.Program() as program:
        pq.Q(0, 2, 5) | pq.StateVector((1, 0, 1))

    # Mode 5 is the highest, so we need d=6
    assert program.get_number_of_modes() == 6


def test_program_get_number_of_modes_empty_program():
    """Test that get_number_of_modes returns None for empty program."""
    with pq.Program() as program:
        pass

    assert program.get_number_of_modes() is None


def test_program_get_number_of_modes_with_multiple_instructions():
    """Test that get_number_of_modes considers all instructions."""
    with pq.Program() as program:
        pq.Q(0) | pq.StateVector((1,))
        pq.Q(1) | pq.StateVector((1,))
        pq.Q(2, 3) | pq.Beamsplitter5050()

    # Mode 3 is the highest, so we need d=4
    assert program.get_number_of_modes() == 4


def test_PureFockSimulator_without_d_parameter_infers_from_program():
    """Test that simulator can infer d from program (main use case)."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))
        pq.Q(0, 1) | pq.Beamsplitter5050()

    simulator = pq.PureFockSimulator()  # No d parameter!
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_PureFockSimulator_with_explicit_d_still_works():
    """Test backward compatibility: explicit d parameter still works."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))
        pq.Q(0, 1) | pq.Beamsplitter5050()

    simulator = pq.PureFockSimulator(d=2)
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_PureFockSimulator_infers_with_sparse_modes():
    """Test that simulator correctly infers d with non-contiguous mode indices."""
    with pq.Program() as program:
        pq.Q(0, 2) | pq.StateVector((1, 1))

    simulator = pq.PureFockSimulator()
    result = simulator.execute(program)

    # Mode 2 is used, so d should be at least 3
    assert result.state.d == 3
    assert simulator.d == 3


def test_PureFockSimulator_without_d_and_empty_program_raises_error():
    """Test that simulator raises error when d cannot be inferred."""
    with pq.Program() as program:
        pass

    simulator = pq.PureFockSimulator()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as exc_info:
        simulator.execute(program)

    assert "Cannot infer the number of modes" in str(exc_info.value)


def test_GaussianSimulator_without_d_parameter_infers_from_program():
    """Test that GaussianSimulator also supports mode inference."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.5, phi=0.3)

    simulator = pq.GaussianSimulator()
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_FockSimulator_without_d_parameter_infers_from_program():
    """Test that FockSimulator also supports mode inference."""
    with pq.Program() as program:
        pq.Q(0, 1) | pq.DensityMatrix(ket=(1, 0), bra=(1, 0))

    simulator = pq.FockSimulator(config=pq.Config(cutoff=3))
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_simulator_d_is_inferred_before_first_execution():
    """Test that d is inferred during execute, not before."""
    simulator = pq.PureFockSimulator()
    assert simulator.d is None

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))

    simulator.execute(program)
    assert simulator.d == 2


def test_simulator_can_execute_multiple_programs_with_same_d():
    """Test that simulator with inferred d can execute multiple programs."""
    simulator = pq.PureFockSimulator()

    # First program
    with pq.Program() as program1:
        pq.Q(0, 1) | pq.StateVector((2, 0))

    result1 = simulator.execute(program1)
    assert result1.state.d == 2

    # Second program with same d
    with pq.Program() as program2:
        pq.Q(0, 1) | pq.StateVector((1, 1))

    result2 = simulator.execute(program2)
    assert result2.state.d == 2


def test_simulator_validation_skips_mode_check_when_d_is_none():
    """Test that validation doesn't fail when d is None."""
    simulator = pq.PureFockSimulator()

    with pq.Program() as program:
        pq.Q(0, 1) | pq.StateVector((2, 0))

    # This should not raise an error even though d is None
    simulator.validate(program)
