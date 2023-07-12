#
# Copyright 2021-2023 Budapest Quantum Computing Group
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
import itertools
import numpy as np
import piquasso as pq


NUMBER_OF_MODES = 3


SIMULATORS = (
    pq.GaussianSimulator,
    pq.FockSimulator,
    pq.PureFockSimulator,
)


SCALAR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=np.pi / 3),
    pq.Fourier(),
    pq.Squeezing(r=0.1, phi=np.pi / 3),
    pq.QuadraticPhase(s=0.3),
    pq.Displacement(r=1, phi=2),
    pq.PositionDisplacement(x=0.4),
    pq.MomentumDisplacement(p=0.2),
)


LENGTH_ONE_VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 3]),
    pq.Fourier(),
    pq.Squeezing(r=[0.1], phi=[np.pi / 3]),
    pq.QuadraticPhase(s=[0.3]),
    pq.Displacement(r=[1], phi=[2]),
    pq.PositionDisplacement(x=[0.4]),
    pq.MomentumDisplacement(p=[0.2]),
)


VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 1, np.pi / 2, np.pi / 3]),
    pq.Fourier(),
    pq.Squeezing(r=[0.1, 0.2, 0.3], phi=[np.pi / 1, np.pi / 2, np.pi / 3]),
    pq.QuadraticPhase(s=[0.1, 0.2, 0.3]),
    pq.Displacement(r=[1, 2, 3], phi=[1j, 2j, 3j]),
    pq.PositionDisplacement(x=[0.4, 0.5, 0.6]),
    pq.MomentumDisplacement(p=[0.2, 0.3, 0.4]),
)


INVALID_VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 1, np.pi / 2]),
    pq.Squeezing(r=[0.1, 0.2], phi=[np.pi / 1, np.pi / 3]),
    pq.QuadraticPhase(s=[0.1, 0.2, 0.4, 0.5, 0.6, 0.8]),
    pq.Displacement(r=[1, 3], phi=[1j, 3j]),
    pq.PositionDisplacement(x=[0.4, 0.6]),
    pq.MomentumDisplacement(p=[0.2, 0.4]),
)


@pytest.mark.parametrize(
    "SimulatorClass, scalar_instruction",
    itertools.product(SIMULATORS, SCALAR_INSTRUCTIONS),
)
def test_scalar_parameters_are_scaled_when_applied(SimulatorClass, scalar_instruction):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | scalar_instruction

    simulator = SimulatorClass(d=NUMBER_OF_MODES)
    result = simulator.execute(program)

    result.state.validate()


@pytest.mark.parametrize(
    "SimulatorClass, length_one_vector_instruction",
    itertools.product(SIMULATORS, LENGTH_ONE_VECTOR_INSTRUCTIONS),
)
def test_length_one_vector_instructions_are_applied_normally(
    SimulatorClass, length_one_vector_instruction
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | length_one_vector_instruction

    simulator = SimulatorClass(d=NUMBER_OF_MODES)
    result = simulator.execute(program)

    result.state.validate()


@pytest.mark.parametrize(
    "SimulatorClass, vector_instruction",
    itertools.product(SIMULATORS, VECTOR_INSTRUCTIONS),
)
def test_vector_instructions_are_applied_normally(SimulatorClass, vector_instruction):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | vector_instruction

    simulator = SimulatorClass(d=NUMBER_OF_MODES)
    result = simulator.execute(program)

    result.state.validate()


@pytest.mark.parametrize(
    "SimulatorClass, invalid_vector_instruction",
    itertools.product(SIMULATORS, INVALID_VECTOR_INSTRUCTIONS),
)
def test_applying_invalid_vector_instructions_raises_error(
    SimulatorClass, invalid_vector_instruction
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | invalid_vector_instruction

    simulator = SimulatorClass(d=NUMBER_OF_MODES)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as excinfo:
        simulator.execute(program)

    assert "is not applicable to modes" in str(excinfo.value)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_cubic_phase_autoscaling_valid(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(all) | pq.CubicPhase(gamma=[0, 1, 2])

    simulator = SimulatorClass(d=3)
    state = simulator.execute(program).state
    state.validate()


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_cubic_phase_autoscaling_invalid(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.CubicPhase(gamma=[0, 1])

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.CubicPhase(gamma=[0, 1, 2])

    simulator = SimulatorClass(d=2)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as excinfo:
        simulator.execute(program_1)

    assert "is not applicable to modes" in str(excinfo.value)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as excinfo:
        simulator.execute(program_2)

    assert "is not applicable to modes" in str(excinfo.value)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_CubicPhase_multimode_equivalence(SimulatorClass):
    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Create() | pq.Create()
        pq.Q(1) | pq.Create()

    with pq.Program() as program_with_multimode_instruction:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.CubicPhase(gamma=[1, 2])

    with pq.Program() as program_with_two_onemode_instructions:
        pq.Q() | preparation

        pq.Q(0) | pq.CubicPhase(gamma=1)
        pq.Q(1) | pq.CubicPhase(gamma=2)

    simulator = SimulatorClass(d=2)

    state_with_multimode_instruction = simulator.execute(
        program_with_multimode_instruction
    ).state
    state_with_two_onemode_instructions = simulator.execute(
        program_with_two_onemode_instructions
    ).state

    assert state_with_multimode_instruction == state_with_two_onemode_instructions


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_kerr_autoscaling_invalid(SimulatorClass):
    with pq.Program() as program_1:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Kerr(xi=[0, 1])

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Kerr(xi=[0, 1, 2])

    simulator = SimulatorClass(d=2)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as excinfo:
        simulator.execute(program_1)

    assert "is not applicable to modes" in str(excinfo.value)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as excinfo:
        simulator.execute(program_2)

    assert "is not applicable to modes" in str(excinfo.value)


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_kerr_autoscaling_valid(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Create() | pq.Create()
        pq.Q(1) | pq.Create()

        pq.Q(all) | pq.Kerr(xi=2)

    simulator = SimulatorClass(d=3)

    state = simulator.execute(program).state

    state.validate()


@pytest.mark.parametrize(
    "SimulatorClass",
    (
        pq.PureFockSimulator,
        pq.FockSimulator,
    ),
)
def test_Kerr_multimode_equivalence(SimulatorClass):
    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Create() | pq.Create()
        pq.Q(1) | pq.Create()

    with pq.Program() as program_with_multimode_instruction:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Kerr(xi=[1, 2])

    with pq.Program() as program_with_two_onemode_instructions:
        pq.Q() | preparation

        pq.Q(0) | pq.Kerr(xi=1)
        pq.Q(1) | pq.Kerr(xi=2)

    simulator = SimulatorClass(d=2)

    state_with_multimode_instruction = simulator.execute(
        program_with_multimode_instruction
    ).state
    state_with_two_onemode_instructions = simulator.execute(
        program_with_two_onemode_instructions
    ).state

    assert state_with_multimode_instruction == state_with_two_onemode_instructions


def test_Kerr_scaling_on_mixed_states():
    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

    with pq.Program() as program_with_multimode_instruction:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Kerr(xi=[-0.1, 0.2])

    with pq.Program() as program_with_two_onemode_instructions:
        pq.Q() | preparation

        pq.Q(0) | pq.Kerr(xi=-0.1)
        pq.Q(1) | pq.Kerr(xi=0.2)

    simulator = pq.FockSimulator(d=2)

    state_with_multimode_instruction = simulator.execute(
        program_with_multimode_instruction
    ).state
    state_with_two_onemode_instructions = simulator.execute(
        program_with_two_onemode_instructions
    ).state

    assert state_with_multimode_instruction == state_with_two_onemode_instructions
