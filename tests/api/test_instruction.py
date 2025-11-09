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

import re

import pytest

from fractions import Fraction

import piquasso as pq

from piquasso.api.exceptions import PiquassoException

from unittest.mock import Mock


def test_registering_instruction_by_subclassing():
    class OtherBeamsplitter(pq.Instruction):
        pass

    assert pq.Instruction.get_subclass("OtherBeamsplitter") is OtherBeamsplitter


def test_subclassing_instruction_with_existing_name_is_successful():
    class Beamsplitter(pq.Instruction):
        pass

    assert pq.Instruction.get_subclass("Beamsplitter") is Beamsplitter
    assert pq.Beamsplitter is not Beamsplitter

    # Teardown
    pq.Instruction.set_subclass(pq.Beamsplitter)
    assert pq.Instruction.get_subclass("Beamsplitter") is pq.Beamsplitter


def test_set_subclass_with_no_subclass():
    any_other_class = object

    with pytest.raises(PiquassoException):
        pq.Instruction.set_subclass(any_other_class)


def test_instruction_when_with_lambda_function():
    class CustomInstruction(pq.Instruction):
        pass

    instruction = CustomInstruction()

    assert instruction.condition is None

    lambda_func = lambda x: x[0] == 1  # noqa: E731

    instruction = instruction.when(lambda_func)

    assert instruction.condition is not None
    assert instruction.condition is lambda_func


def test_instruction_when_called_twice_raises_exception():
    class CustomInstruction(pq.Instruction):
        pass

    instruction = CustomInstruction()

    lambda_func = lambda x: x[0] == 1  # noqa: E731

    instruction = instruction.when(lambda_func)

    with pytest.raises(
        PiquassoException,
        match=re.escape(
            f"The instruction '{instruction}' is already conditioned on "
            f"'{lambda_func}'."
        ),
    ):
        instruction.when(lambda_func)


def test_instruction_when_with_expression_string():
    class CustomInstruction(pq.Instruction):
        pass

    instruction = CustomInstruction()

    non_callable_condition = "x[0] == 1"

    instruction = instruction.when(non_callable_condition)

    assert instruction.condition is not None
    assert str(instruction.condition) == non_callable_condition


def test_instruction_condition_evaluation_with_exception_raises_piquasso_exception(
    FakeState, FakeConfig, FakeConnector
):
    class FakeMeasurement(pq.Measurement):
        pass

    def fake_measurement(state: FakeState, instruction: pq.Instruction, shots: int):
        return [
            pq.api.branch.Branch(state=state, frequency=Fraction(3, 10), outcome=(0,)),
            pq.api.branch.Branch(state=state, frequency=Fraction(7, 10), outcome=(1,)),
        ]

    class CustomInstruction(pq.Instruction):
        pass

    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        return [pq.api.branch.Branch(state=state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            CustomInstruction: fake_calculation,
            FakeMeasurement: fake_measurement,
        }

        _default_connector_class = FakeConnector

        _measurement_classes_allowed_mid_circuit = FakeMeasurement

    instruction = CustomInstruction()

    def faulty_condition(outcomes):
        return outcomes[0] / 0  # This will raise a ZeroDivisionError

    simulator = FakeSimulator(d=1)

    instruction = instruction.when(faulty_condition)

    with pytest.raises(PiquassoException) as exc:
        simulator.execute_instructions([FakeMeasurement(), instruction], shots=10)

    assert exc.value.args[0] == (
        f"An error occurred when evaluating the condition '{faulty_condition}' for the "
        f"instruction '{instruction}': division by zero\nMake sure that the condition "
        f"is a callable that takes the tuple of measurement outcomes and returns a "
        f"boolean, OR an expression string, which can be evaluated to a boolean."
    )


def test_instruction_condition_expression_evaluation_with_exception_raises_piquasso_exception(  # noqa: E501
    FakeState, FakeConfig, FakeConnector
):
    class FakeMeasurement(pq.Measurement):
        pass

    def fake_measurement(state: FakeState, instruction: pq.Instruction, shots: int):
        return [
            pq.api.branch.Branch(state=state, frequency=Fraction(3, 10), outcome=(0,)),
            pq.api.branch.Branch(state=state, frequency=Fraction(7, 10), outcome=(1,)),
        ]

    class CustomInstruction(pq.Instruction):
        pass

    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        return [pq.api.branch.Branch(state=state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            CustomInstruction: fake_calculation,
            FakeMeasurement: fake_measurement,
        }

        _default_connector_class = FakeConnector

        _measurement_classes_allowed_mid_circuit = FakeMeasurement

    instruction = CustomInstruction()

    simulator = FakeSimulator(d=1)

    instruction = instruction.when("x[0] / 0 == 1")

    with pytest.raises(PiquassoException) as exc:
        simulator.execute_instructions([FakeMeasurement(), instruction], shots=10)

    assert exc.value.args[0] == (
        f"An error occurred when evaluating the condition 'x[0] / 0 == 1' for the "
        f"instruction '{instruction}': division by zero\nMake sure that the condition "
        f"is a callable that takes the tuple of measurement outcomes and returns a "
        f"boolean, OR an expression string, which can be evaluated to a boolean."
    )


def test_instruction_unresolved_parameter(FakeState, FakeConfig, FakeConnector):
    class FakeMeasurement(pq.Measurement):
        pass

    def fake_measurement(state: FakeState, instruction: pq.Instruction, shots: int):
        return [
            pq.api.branch.Branch(state=state, frequency=Fraction(3, 10), outcome=(0,)),
            pq.api.branch.Branch(state=state, frequency=Fraction(7, 10), outcome=(1,)),
        ]

    class CustomInstruction(pq.Instruction):
        def __init__(self, param):
            super().__init__(params=dict(param=param))

    zero_branch_state = Mock()
    one_branch_state = Mock()

    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        if instruction.params["param"] == 0:
            return [pq.api.branch.Branch(state=zero_branch_state)]
        else:
            return [pq.api.branch.Branch(state=one_branch_state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            CustomInstruction: fake_calculation,
            FakeMeasurement: fake_measurement,
        }

        _default_connector_class = FakeConnector

        _measurement_classes_allowed_mid_circuit = FakeMeasurement

    instruction = CustomInstruction(param=lambda outcomes: outcomes[-1])

    simulator = FakeSimulator(d=1)

    result = simulator.execute_instructions([FakeMeasurement(), instruction], shots=10)

    for branch in result.branches:
        if branch.outcome == (0,):
            assert branch.state == zero_branch_state
        elif branch.outcome == (1,):
            assert branch.state == one_branch_state


def test_instruction_faulty_unresolved_parameter_raises_InvalidParameter(
    FakeState, FakeConfig, FakeConnector
):
    class FakeMeasurement(pq.Measurement):
        pass

    def fake_measurement(state: FakeState, instruction: pq.Instruction, shots: int):
        return [
            pq.api.branch.Branch(state=state, frequency=Fraction(3, 10), outcome=(0,)),
            pq.api.branch.Branch(state=state, frequency=Fraction(7, 10), outcome=(1,)),
        ]

    class CustomInstruction(pq.Instruction):
        def __init__(self, param):
            super().__init__(params=dict(param=param))

    zero_branch_state = Mock()
    one_branch_state = Mock()

    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        if instruction.params["param"] == 0:
            return [pq.api.branch.Branch(state=zero_branch_state)]
        else:
            return [pq.api.branch.Branch(state=one_branch_state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            CustomInstruction: fake_calculation,
            FakeMeasurement: fake_measurement,
        }

        _default_connector_class = FakeConnector

        _measurement_classes_allowed_mid_circuit = FakeMeasurement

    instruction = CustomInstruction(param=lambda outcomes: outcomes[-1] / 0)

    simulator = FakeSimulator(d=1)

    with pytest.raises(PiquassoException) as exc:
        simulator.execute_instructions([FakeMeasurement(), instruction], shots=10)

    assert exc.value.args[0] == (
        f"An error occurred when resolving the parameter 'param' for the instruction "
        f"'{instruction}': division by zero\n"
        "Make sure that the parameter is a callable that takes the tuple of "
        "measurement outcomes and returns a float, OR an expression string, which can "
        "be evaluated to a float."
    )


def test_instruction_faulty_unresolved_expression_parameter_raises_InvalidParameter(
    FakeState, FakeConfig, FakeConnector
):
    class FakeMeasurement(pq.Measurement):
        pass

    def fake_measurement(state: FakeState, instruction: pq.Instruction, shots: int):
        return [
            pq.api.branch.Branch(state=state, frequency=Fraction(3, 10), outcome=(0,)),
            pq.api.branch.Branch(state=state, frequency=Fraction(7, 10), outcome=(1,)),
        ]

    class CustomInstruction(pq.Instruction):
        def __init__(self, param):
            super().__init__(params=dict(param=param))

    zero_branch_state = Mock()
    one_branch_state = Mock()

    def fake_calculation(state: FakeState, instruction: pq.Instruction, shots: int):
        if instruction.params["param"] == 0:
            return [pq.api.branch.Branch(state=zero_branch_state)]
        else:
            return [pq.api.branch.Branch(state=one_branch_state)]

    class FakeSimulator(pq.Simulator):
        _state_class = FakeState

        _config_class = FakeConfig

        _instruction_map = {
            CustomInstruction: fake_calculation,
            FakeMeasurement: fake_measurement,
        }

        _default_connector_class = FakeConnector

        _measurement_classes_allowed_mid_circuit = FakeMeasurement

    instruction = CustomInstruction(param="x[-1] / 0")

    simulator = FakeSimulator(d=1)

    with pytest.raises(PiquassoException) as exc:
        simulator.execute_instructions([FakeMeasurement(), instruction], shots=10)

    assert exc.value.args[0] == (
        f"An error occurred when resolving the parameter 'param' for the instruction "
        f"'{instruction}': division by zero\n"
        "Make sure that the parameter is a callable that takes the tuple of "
        "measurement outcomes and returns a float, OR an expression string, which can "
        "be evaluated to a float."
    )
