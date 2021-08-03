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
import piquasso as pq

from functools import partial

CUTOFF = 4


@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_InvalidModes_are_raised_if_modes_are_already_measured(StateClass):
    with pytest.raises(pq.api.errors.InvalidProgram):
        with pq.Program():
            pq.Q() | pq.Vacuum()

            pq.Q(0, 1) | pq.ParticleNumberMeasurement()

            pq.Q(2) | pq.ParticleNumberMeasurement()
