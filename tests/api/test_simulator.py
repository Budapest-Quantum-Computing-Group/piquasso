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


def test_program_execution_with_invalid_modes_raises_InvalidModes(
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
