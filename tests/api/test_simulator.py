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

from unittest.mock import Mock


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


def test_conditional_instruction_execution(
    FakeState,
    FakeConfig,
    FakeConnector,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    def fake_calculation(state, instruction, shots):
        return [pq.api.branch.Branch(state=state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            FakePreparation: fake_calculation,
            FakeGate: Mock(
                return_value=[
                    pq.api.branch.Branch(
                        state=FakeState(1, FakeConnector(), FakeConfig())
                    )
                ]
            ),
            FakeMeasurement: fake_calculation,
        }

        _default_connector_class = FakeConnector

    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate().when(lambda x: True)
        pq.Q() | FakeMeasurement()

    simulator = FakeSimulator(d=1)

    simulator.validate(program)
    simulator.execute(program)

    FakeSimulator._instruction_map[FakeGate].assert_called_once()

    with pq.Program() as new_program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate().when(lambda x: False)
        pq.Q() | FakeMeasurement()

    simulator.validate(new_program)
    simulator.execute(new_program)

    FakeSimulator._instruction_map[FakeGate].assert_called_once()


def test_simulator_without_d_parameter_infers_from_program(
    FakeSimulator, FakePreparation, FakeGate
):
    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()
        pq.Q(0, 1) | FakeGate()

    simulator = FakeSimulator()
    result = simulator.execute(program)

    assert result.state.d == 2
    assert simulator.d is None

    with pq.Program() as another_program:
        pq.Q(0, 1) | FakePreparation()
        pq.Q(2, 3) | FakeGate()

    another_result = simulator.execute(another_program)

    assert another_result.state.d == 4
    assert simulator.d is None


def test_simulator_with_d_parameter_uses_given_d(
    FakeSimulator, FakePreparation, FakeGate
):
    with pq.Program() as program:
        pq.Q(0, 1) | FakePreparation()
        pq.Q(0, 1) | FakeGate()

    simulator = FakeSimulator(d=5)
    result = simulator.execute(program)

    assert result.state.d == 5
    assert simulator.d == 5


def test_simulator_execution_without_d_parameter_and_no_modes_instructions_raises_InvalidSimulation(  # noqa: E501
    FakeSimulator,
    FakePreparation,
):
    with pq.Program() as program:
        pq.Q() | FakePreparation()

    simulator = FakeSimulator()

    with pytest.raises(pq.api.exceptions.InvalidSimulation) as error:
        simulator.execute(program)

    error_message = error.value.args[0]

    assert (
        error_message
        == "The number of modes could not be inferred from the instructions, and 'd' "
        "was not specified during simulator initialization. Please provide the number "
        "of modes 'd' when creating the simulator, or ensure that the instructions "
        "address specific modes."
    )


def test_create_initial_state(FakeSimulator, FakeState):
    simulator = FakeSimulator(d=3)

    initial_state = simulator.create_initial_state()

    assert isinstance(initial_state, FakeState)
    assert initial_state.d == 3


def test_create_initial_state_with_d_parameter(FakeSimulator, FakeState):
    simulator = FakeSimulator()

    initial_state = simulator.create_initial_state(d=4)

    assert isinstance(initial_state, FakeState)
    assert initial_state.d == 4


def test_create_initial_state_without_d_raises_InvalidParameter(FakeSimulator):
    simulator = FakeSimulator()

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.create_initial_state()

    error_message = error.value.args[0]

    assert (
        error_message
        == "The number of modes 'd' must be specified to create the initial state. "
        "Please provide 'd' when creating the simulator, or pass 'd' as an argument "
        "to 'create_initial_state'."
    )


def test_shots_none_with_measurement_allowed(
    FakeSimulator,
    FakePreparation,
    FakeGate,
    FakeMeasurement,
):
    class MeasurementAllowedWithShotsNone(FakeMeasurement):
        pass

    with pq.Program() as program:
        pq.Q() | FakePreparation()
        pq.Q() | FakeGate()
        pq.Q() | MeasurementAllowedWithShotsNone()

    class SimulatorWithMeasurementAllowedWithShotsNone(FakeSimulator):
        _instruction_map = {
            **FakeSimulator._instruction_map,
            MeasurementAllowedWithShotsNone: lambda state, instruction, shots: [],
        }

        _measurement_classes_allowed_with_shots_none = (
            MeasurementAllowedWithShotsNone,
        )

    simulator = SimulatorWithMeasurementAllowedWithShotsNone(d=1)

    result = simulator.execute(program, shots=None)

    assert isinstance(result, pq.api.result.Result)


def test_shots_none_with_measurement_not_allowed_raises_InvalidParameter(
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

    with pytest.raises(
        pq.api.exceptions.InvalidParameter,
        match=(
            "The measurement 'FakeMeasurement' instruction does not support "
            "'shots=None' using 'FakeSimulator'."
        ),
    ):
        simulator.execute(program, shots=None)


def test_shots_with_invalid_value_raises_InvalidParameter(
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

    with pytest.raises(
        pq.api.exceptions.InvalidParameter,
        match="The number of shots should be a positive integer or 'None': shots=0.",
    ):
        simulator.execute(program, shots=0)
