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
import re

import piquasso as pq


def test_correctly_defined_program_executes_without_exception(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    simulator.validate(program)
    simulator.execute(program)


def test_program_defined_without_measurement_executes_without_exception(
    FakeSimulator,
    FakePreparation,
    FakeGate,
):
    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()

    simulator = FakeSimulator(d=1)

    simulator.validate(program)
    simulator.execute(program)


def test_program_defined_with_multiple_gates_or_preparations_executes_without_exception(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakePreparation()
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()
        pq.Q() | FakeGate()
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    simulator.validate(program)
    simulator.execute(program)


def test_empty_program_executes_without_exception(
    FakeSimulator,
):
    with pq.Program() as program:
        pass

    simulator = FakeSimulator(d=1)

    simulator.validate(program)
    simulator.execute(program)


def test_program_execution_with_unregistered_instruction_raises_InvalidSimulation(
    FakeSimulator,
    FakeGate,
    FakeMeasurement,
):
    class ImproperInstruction(pq.Instruction):
        pass

    with pq.Program() as program:
        pq.Q() | ImproperInstruction()
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        simulator.validate(program)

    error_message = error.value.args[0]

    assert "No such instruction implemented for this simulator." in error_message
    assert "instruction=ImproperInstruction(modes=())" in error_message

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        simulator.execute(program)

    error_message = error.value.args[0]

    assert "No such instruction implemented for this simulator." in error_message
    assert "instruction=ImproperInstruction(modes=())" in error_message


def test_program_execution_with_wrong_instruction_order_raises_InvalidSimulation(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakeGate()
        pq.Q() | FakePreparation()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.execute(program)


def test_program_execution_with_duplicated_blocks_raises_InvalidSimulation(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()

        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.execute(program)


def test_program_execution_with_multiple_measurements_raises_InvalidSimulation(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakeGate()
        pq.Q() | FakePreparation()
        pq.Q() | FakeMeasurement()
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.execute(program)


def test_program_execution_with_measurement_not_being_last_raises_InvalidSimulation(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    with pq.Program() as program:
        pq.Q() | FakeGate()
        pq.Q() | FakeMeasurement()
        pq.Q() | FakePreparation()

    simulator = FakeSimulator(d=1)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidSimulation):
        simulator.execute(program)


def test_program_execution_with_initial_state(
    FakeSimulator,
    FakeGate,
    FakeState,
):
    simulator = FakeSimulator(d=1)

    with pq.Program() as program:
        pass

    initial_state = simulator.execute(program).state

    with pq.Program() as program:
        pq.Q() | FakeGate()

    simulator.validate(program)
    simulator.execute(program, initial_state=initial_state)


def test_program_execution_with_initial_state_of_wrong_type_raises_InvalidState(
    FakeSimulator,
    FakeState,
):
    simulator = FakeSimulator(d=1)

    with pq.Program() as program:
        pass

    simulator.validate(program)

    initial_state = simulator.execute(program).state

    class OtherFakeState(FakeState):
        pass

    class OtherFakeSimulator(FakeSimulator):
        _state_class = OtherFakeState

    other_simulator = OtherFakeSimulator(d=1)

    other_simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidState) as exc:
        other_simulator.execute(program, initial_state=initial_state)

    assert "Initial state is specified with type" in exc.value.args[0]


def test_program_execution_with_initial_state_of_wrong_no_of_modes_raises_InvalidState(
    FakeSimulator,
):
    single_mode_simulator = FakeSimulator(d=1)

    with pq.Program() as program:
        pass

    single_mode_simulator.validate(program)

    single_mode_initial_state = single_mode_simulator.execute(program).state

    two_mode_simulator = FakeSimulator(d=2)

    two_mode_simulator.validate(program)

    expected_error_message = (
        "Mismatch in number of specified modes: According to the simulator, "
        "the number of modes should be '2', but the specified 'initial_state' "
        "is defined on '1' modes."
    )

    with pytest.raises(pq.api.exceptions.InvalidState, match=expected_error_message):
        two_mode_simulator.execute(program, initial_state=single_mode_initial_state)


def test_program_execution_with_invalid_modes_raises_InvalidModes_with_single_mode(
    FakeSimulator,
    FakeGate,
):
    with pq.Program() as program:
        pq.Q(0, 1) | FakeGate()

    simulator = FakeSimulator(d=1)

    expected_error_message = re.escape(
        "Instruction 'FakeGate(modes=(0, 1))' addresses mode '1', "
        "which is out of range for the simulator defined on '1' modes. "
        "For a single-mode system, the only valid mode index is '0'."
    )

    with pytest.raises(pq.api.exceptions.InvalidModes, match=expected_error_message):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidModes, match=expected_error_message):
        simulator.execute(program)


def test_program_execution_with_invalid_modes_raises_InvalidModes_with_multiple_modes(
    FakeSimulator,
    FakeGate,
):
    with pq.Program() as program:
        pq.Q(0, 2) | FakeGate()

    simulator = FakeSimulator(d=2)

    expected_error_message = re.escape(
        "Instruction 'FakeGate(modes=(0, 2))' addresses mode '2', "
        "which is out of range for the simulator defined on '2' modes. "
        "Valid mode indices are between '0' and '1' (inclusive)."
    )

    with pytest.raises(pq.api.exceptions.InvalidModes, match=expected_error_message):
        simulator.validate(program)

    with pytest.raises(pq.api.exceptions.InvalidModes, match=expected_error_message):
        simulator.execute(program)


def test_Config_override(FakeSimulator, FakeConfig):
    with pq.Program() as program:
        pass

    simulator = FakeSimulator(d=1)
    state = simulator.execute(program).state

    assert isinstance(simulator.config, FakeConfig)
    assert isinstance(state._config, FakeConfig)


def test_Simulator_repr(FakeSimulator):
    assert (
        repr(FakeSimulator(d=3))
        == "FakeSimulator(d=3, config=Config(), connector=FakeConnector())"
    )


def test_simulator_without_d_parameter_infers_from_program(
    FakeSimulator, FakePreparation, FakeGate
):
    """Test that simulator can infer d from program when not provided."""
    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()
        pq.Q(0, 1) | FakeGate()

    simulator = FakeSimulator()  # No d parameter!
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_simulator_with_explicit_d_still_works(
    FakeSimulator, FakePreparation, FakeGate
):
    """Test backward compatibility: explicit d parameter still works."""
    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()
        pq.Q(0, 1) | FakeGate()

    simulator = FakeSimulator(d=2)
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d == 2


def test_simulator_infers_with_sparse_modes(FakeSimulator, FakePreparation):
    """Test that simulator correctly infers d with non-contiguous mode indices."""
    with pq.Program() as program:
        pq.Q(0, 2) | FakePreparation()

    simulator = FakeSimulator()
    result = simulator.execute(program)

    # Mode 2 is used, so d should be at least 3
    assert result.state.d == 3
    assert simulator.d == 3


def test_simulator_without_d_and_empty_program_raises_error(FakeSimulator):
    """Test that simulator raises error when d cannot be inferred."""
    with pq.Program() as program:
        pass

    simulator = FakeSimulator()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as exc_info:
        simulator.execute(program)

    assert "Cannot infer the number of modes" in str(exc_info.value)


def test_simulator_d_is_inferred_during_execution(FakeSimulator, FakePreparation):
    """Test that d is inferred during execute, not before."""
    simulator = FakeSimulator()
    assert simulator.d is None

    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()

    simulator.execute(program)
    assert simulator.d == 2


def test_simulator_can_execute_multiple_programs_with_same_d(
    FakeSimulator, FakePreparation
):
    """Test that simulator with inferred d can execute multiple programs."""
    simulator = FakeSimulator()

    # First program
    with pq.Program() as program1:
        pq.Q(0, 1) | FakePreparation()

    result1 = simulator.execute(program1)
    assert result1.state.d == 2

    # Second program with same d
    with pq.Program() as program2:
        pq.Q(0, 1) | FakePreparation()

    result2 = simulator.execute(program2)
    assert result2.state.d == 2


def test_simulator_execute_instructions_without_program_raises_error(
    FakeSimulator, FakePreparation
):
    """Test that execute_instructions requires d to be set or program to be provided."""
    simulator = FakeSimulator()

    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()

    # execute_instructions without d and without program should fail
    with pytest.raises(pq.api.exceptions.InvalidSimulation) as exc_info:
        simulator.execute_instructions(program.instructions)

    assert "Cannot infer the number of modes" in str(exc_info.value)
