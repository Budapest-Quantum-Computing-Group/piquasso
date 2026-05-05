#
# Copyright 2021-2026 Budapest Quantum Computing Group
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


def test_simulate_Gaussian_instructions():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.2, phi=np.pi / 4)

    result = pq.simulate(program)

    assert type(result) is pq.api.result.Result

    assert type(result.state) is pq.GaussianState, (
        "The state should be a GaussianState, as the program only contains "
        "instructions compatible with Gaussian states."
    )


def test_simulate_passive_linear_gates():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])

        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

    result = pq.simulate(program, connector=pq.JaxConnector())

    assert type(result.state) is pq.SamplingState, (
        "The state should be a SamplingState, as it only uses passive linear optical "
        "gates."
    )


def test_simulate_nonlinear_gates():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])

        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(0) | pq.Kerr(np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter5050()

    result = pq.simulate(program, config=pq.Config(cutoff=9))

    assert (
        type(result.state) is pq.PureFockState
    ), "The state should be a PureFockState, as it uses nonlinear gates."


def test_simulate_nonlinear_gates_and_channels():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])

        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(0) | pq.Kerr(np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0) | pq.Attenuator(theta=0.1)

    result = pq.simulate(program, number_of_modes=3)

    assert (
        type(result.state) is pq.FockState
    ), "The state should be a FockState, as it uses nonlinear gates and channels."


def test_simulate_raises_NotImplementedCalculation_when_the_instruction_combination_is_not_supported():  # noqa: E501
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])

        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(0) | pq.Kerr(np.pi / 4)
        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0) | pq.DeterministicGaussianChannel(X=np.eye(3), Y=np.eye(3))

    with pytest.raises(pq.api.exceptions.NotImplementedCalculation) as exc:
        pq.simulate(program)

    assert exc.value.args[0] == (
        "The program cannot be simulated by any of the built-in simulators in "
        "Piquasso. Please verify that the program's instructions are supported by at "
        "least one built-in simulator. For requests to implement a new simulation "
        "pathway, please open an issue on the Piquasso GitHub repository: "
        "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
    )


def test_simulate_raises_NotImplementedCalculation_when_connector_is_not_supported():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 0])

        pq.Q(0, 1) | pq.Beamsplitter5050()
        pq.Q(1, 2) | pq.Beamsplitter5050()

    not_even_a_connector = object()

    with pytest.raises(pq.api.exceptions.NotImplementedCalculation) as exc:
        pq.simulate(program, connector=not_even_a_connector)

    assert exc.value.args[0] == (
        "The specified program cannot be simulated by the connector "
        "'object'. Please consider using another connector from "
        "'JaxConnector, NumpyConnector, TensorflowConnector', or open an issue on the "
        "Piquasso GitHub repository: "
        "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
    )
