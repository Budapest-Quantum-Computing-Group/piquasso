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


@pytest.mark.parametrize("StateClass", (pq.FockState, pq.PNCFockState))
def test_FockState_reduced(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1/8)

    program.execute()

    with pq.Program() as reduced_program:
        pq.Q() | pq.FockState(d=1, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(1, ), bra=(1, )) / 4

        pq.Q() | pq.DensityMatrix(ket=(2, ), bra=(2, )) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, ), bra=(0, )) / 2

    reduced_program.execute()

    expected_reduced_state = reduced_program.state

    reduced_state = program.state.reduced(modes=(1, ))

    assert expected_reduced_state == reduced_state
