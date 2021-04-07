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

import numpy as np

import piquasso as pq


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass", (pq.PureFockState, pq.PNCFockState, pq.FockState)
)
def test_squeezing_probabilities(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.5, phi=np.pi / 3)

    program.execute()
    program.state.validate()

    assert np.isclose(
        sum(program.state.fock_probabilities),
        1.0
    ), "The state should be renormalized for probability conservation."

    assert np.allclose(
        program.state.fock_probabilities,
        [0.90352508, 0, 0, 0, 0, 0.09647492]
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass", (pq.PureFockState, pq.PNCFockState, pq.FockState)
)
def test_displacement_probabilities(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5, phi=np.pi / 3)

    program.execute()
    program.state.validate()

    assert np.isclose(
        sum(program.state.fock_probabilities),
        1.0
    ), "The state should be renormalized for probability conservation."

    assert np.allclose(
        program.state.fock_probabilities,
        [0.7804878, 0, 0.19512195, 0, 0, 0.02439024]
    )


def test_PureFockState_squeezing():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.5, phi=np.pi / 3)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 2

    assert np.isclose(nonzero_elements[0][0], 0.9505393652547215)
    assert nonzero_elements[0][1] == (0, 0)

    assert np.isclose(nonzero_elements[1][0], -0.15530205657134022+0.26899105250149724j)
    assert nonzero_elements[1][1] == (2, 0)


def test_PureFockState_displacement():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5, phi=np.pi / 3)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 3

    assert np.isclose(nonzero_elements[0][0], 0.8834522085987724)
    assert nonzero_elements[0][1] == (0, 0)

    assert np.isclose(nonzero_elements[1][0], -0.22086305214969315+0.38254602783800296j)
    assert nonzero_elements[1][1] == (1, 0)

    assert np.isclose(nonzero_elements[2][0], -0.07808688094430298-0.13525044520011487j)
    assert nonzero_elements[2][1] == (2, 0)


def test_PNCFockState_squeezing():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.5, phi=np.pi / 3)

    with pytest.warns(UserWarning):
        program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 2

    assert np.isclose(nonzero_elements[0][0], 0.903525084898849)
    assert nonzero_elements[0][1] == ((0, 0), (0, 0))

    assert np.isclose(nonzero_elements[1][0], 0.09647491510115103)
    assert nonzero_elements[1][1] == ((2, 0), (2, 0))


def test_PNCFockState_displacement():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState(d=2, cutoff=2) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5, phi=np.pi / 3)

    with pytest.warns(UserWarning):
        program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 2

    assert np.isclose(nonzero_elements[0][0], 0.8)
    assert nonzero_elements[0][1] == ((0, 0), (0, 0))

    assert np.isclose(nonzero_elements[1][0], 0.2)
    assert nonzero_elements[1][1] == ((1, 0), (1, 0))


def test_FockState_squeezing():
    with pq.Program() as program:
        pq.Q() | pq.FockState(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.5, phi=np.pi / 3)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 4

    assert np.isclose(nonzero_elements[0][0], 0.903525084898849)
    assert nonzero_elements[0][1] == ((0, 0), (0, 0))

    assert np.isclose(nonzero_elements[1][0], -0.14762071827607462-0.2556865843039727j)
    assert nonzero_elements[1][1] == ((0, 0), (2, 0))

    assert np.isclose(nonzero_elements[2][0], -0.14762071827607462+0.2556865843039727j)
    assert nonzero_elements[2][1] == ((2, 0), (0, 0))

    assert np.isclose(nonzero_elements[3][0], 0.09647491510115103)
    assert nonzero_elements[3][1] == ((2, 0), (2, 0))


def test_FockState_displacement():
    with pq.Program() as program:
        pq.Q() | pq.FockState(d=2, cutoff=2) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=0.5, phi=np.pi / 3)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 4

    assert np.isclose(nonzero_elements[0][0], 0.8)
    assert nonzero_elements[0][1] == ((0, 0), (0, 0))

    assert np.isclose(nonzero_elements[1][0], -0.2 - 0.3464101615137754j)
    assert nonzero_elements[1][1] == ((0, 0), (1, 0))

    assert np.isclose(nonzero_elements[2][0], -0.2 + 0.3464101615137754j)
    assert nonzero_elements[2][1] == ((1, 0), (0, 0))

    assert np.isclose(nonzero_elements[3][0], 0.2)
    assert nonzero_elements[3][1] == ((1, 0), (1, 0))
