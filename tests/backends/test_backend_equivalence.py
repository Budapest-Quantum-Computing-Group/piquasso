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

from functools import partial


CUTOFF = 4


def is_proportional(first, second):
    first = np.array(first)
    second = np.array(second)

    index = np.argmax(first)

    proportion = first[index] / second[index]

    return np.allclose(first, proportion * second)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_squeezed_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=0.6)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.99502075,
            0., 0., 0.,
            0., 0., 0., 0., 0., 0.00494212,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "StateClass",
    (
        pq.GaussianState,
        partial(pq.PureFockState, cutoff=CUTOFF),
        partial(pq.FockState, cutoff=CUTOFF),
        partial(pq.PNCFockState, cutoff=CUTOFF),
    )
)
def test_get_fock_probabilities_with_displaced_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(alpha=1 + 2j)

    program.execute()

    probabilities = program.state.get_fock_probabilities(cutoff=CUTOFF)

    assert all(probability >= 0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)

    assert is_proportional(
        probabilities,
        [
            0.00673795,
            0., 0., 0.03368973,
            0., 0., 0., 0., 0., 0.08422434,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1403739,
        ]
    )
