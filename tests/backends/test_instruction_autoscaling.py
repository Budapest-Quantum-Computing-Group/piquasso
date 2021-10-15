#
# Copyright 2021 Budapest Quantum Computing Group
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


STATES = (
    pq.GaussianState,
    pq.PureFockState,
    pq.FockState,
)


SCALAR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=np.pi / 3),
    pq.Fourier(),
    pq.Squeezing(r=0.1, phi=np.pi / 3),
    pq.QuadraticPhase(s=0.3),
    pq.Displacement(alpha=1 + 2j),
    pq.Displacement(r=1, phi=2j),
    pq.PositionDisplacement(x=0.4),
    pq.MomentumDisplacement(p=0.2),
)


LENGTH_ONE_VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 3]),
    pq.Fourier(),
    pq.Squeezing(r=[0.1], phi=[np.pi / 3]),
    pq.QuadraticPhase(s=[0.3]),
    pq.Displacement(alpha=[1 + 2j]),
    pq.Displacement(r=[1], phi=[2j]),
    pq.PositionDisplacement(x=[0.4]),
    pq.MomentumDisplacement(p=[0.2]),
)


VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 1, np.pi / 2, np.pi / 3]),
    pq.Fourier(),
    pq.Squeezing(r=[0.1, 0.2, 0.3], phi=[np.pi / 1, np.pi / 2, np.pi / 3]),
    pq.QuadraticPhase(s=[0.1, 0.2, 0.3]),
    pq.Displacement(alpha=[1 + 2j, -1 + 2j, -2 - 1j]),
    pq.Displacement(r=[1, 2, 3], phi=[1j, 2j, 3j]),
    pq.PositionDisplacement(x=[0.4, 0.5, 0.6]),
    pq.MomentumDisplacement(p=[0.2, 0.3, 0.4]),
)


INVALID_VECTOR_INSTRUCTIONS = (
    pq.Phaseshifter(phi=[np.pi / 1, np.pi / 2]),
    pq.Squeezing(r=[0.1, 0.2], phi=[np.pi / 1, np.pi / 3]),
    pq.QuadraticPhase(s=[0.1, 0.2, 0.4, 0.5, 0.6, 0.8]),
    pq.Displacement(alpha=[1 + 2j, -2 - 1j]),
    pq.Displacement(r=[1, 3], phi=[1j, 3j]),
    pq.PositionDisplacement(x=[0.4, 0.6]),
    pq.MomentumDisplacement(p=[0.2, 0.4]),
)


@pytest.mark.parametrize(
    "StateClass, scalar_instruction",
    itertools.product(STATES, SCALAR_INSTRUCTIONS)
)
def test_scalar_parameters_are_scaled_when_applied(StateClass, scalar_instruction):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | scalar_instruction

    state = StateClass(d=NUMBER_OF_MODES)
    state.apply(program)

    state.validate()


@pytest.mark.parametrize(
    "StateClass, length_one_vector_instruction",
    itertools.product(STATES, LENGTH_ONE_VECTOR_INSTRUCTIONS)
)
def test_length_one_vector_instructions_are_applied_normally(
    StateClass, length_one_vector_instruction
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | length_one_vector_instruction

    state = StateClass(d=NUMBER_OF_MODES)
    state.apply(program)

    state.validate()


@pytest.mark.parametrize(
    "StateClass, vector_instruction",
    itertools.product(STATES, VECTOR_INSTRUCTIONS)
)
def test_vector_instructions_are_applied_normally(StateClass, vector_instruction):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | vector_instruction

    state = StateClass(d=NUMBER_OF_MODES)
    state.apply(program)

    state.validate()


@pytest.mark.parametrize(
    "StateClass, invalid_vector_instruction",
    itertools.product(STATES, INVALID_VECTOR_INSTRUCTIONS)
)
def test_applying_invalid_vector_instructions_raises_error(
    StateClass, invalid_vector_instruction
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | invalid_vector_instruction

    state = StateClass(d=NUMBER_OF_MODES)

    with pytest.raises(pq.api.errors.InvalidParameter) as excinfo:
        state.apply(program)

    assert "is not applicable to modes" in str(excinfo.value)
