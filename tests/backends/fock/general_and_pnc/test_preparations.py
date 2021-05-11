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


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_number_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_and_annihilate_number_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [1, 0, 0, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_annihilate_and_create(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3) | pq.Vacuum()

        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_overflow_with_zero_norm_raises_InvalidState(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 2/5
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0)) * 3/5

        pq.Q(1, 2) | pq.Create()

    with pytest.raises(pq.api.errors.InvalidState) as error:
        program.execute()

    assert error.value.args[0] == "The norm of the state is 0."


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_creation_on_multiple_modes(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=4)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 2/5
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0)) * 3/5

        pq.Q(1, 2) | pq.Create()

    program.execute()

    assert np.isclose(program.state.norm, 1)

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 2/5, 3/5, 0, 0, 0, 0, 0, 0, 0
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_state_is_renormalized_after_overflow(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | (2/6) * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1))
        pq.Q() | (3/6) * pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0))
        pq.Q() | (1/6) * pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))

        pq.Q(2) | pq.Create()

    program.execute()

    assert np.isclose(program.state.norm, 1)

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0, 0, 0,
            0.4, 0.6, 0, 0, 0, 0
        ],
    )
