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

import numpy as np

import piquasso as pq

import pytest

import re


def test_Loss_uniform():
    d = 5
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])

        for i in range(5):
            pq.Q(i) | pq.Loss(0.9)

    simulator = pq.PassiveSimulator(d=d)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy

    singular_values = np.linalg.svd(state.interferometer)[1]

    assert np.allclose(singular_values, [0.9, 0.9, 0.9, 0.9, 0.9])


def test_Loss_non_uniform():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])

        pq.Q(0) | pq.Loss(transmissivity=0.4)
        pq.Q(1) | pq.Loss(transmissivity=0.5)

    simulator = pq.PassiveSimulator(d=5)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy

    singular_values = np.linalg.svd(state.interferometer)[1]

    assert len(singular_values[np.isclose(singular_values, 0.4)]) == 1
    assert len(singular_values[np.isclose(singular_values, 0.5)]) == 1
    assert len(singular_values[np.isclose(singular_values, 1.0)]) == 3


def test_UniformLoss():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])

        pq.Q() | pq.UniformLoss(0.8)

    simulator = pq.PassiveSimulator(d=5)
    state = simulator.execute(program, shots=1).state

    assert state.is_lossy

    singular_values = np.linalg.svd(state.interferometer)[1]

    assert np.allclose(singular_values, [0.8, 0.8, 0.8, 0.8, 0.8])


def test_UniformLoss_transmissivity_out_of_range():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])

        pq.Q() | pq.UniformLoss(-0.1)

    with pytest.raises(
        pq.api.exceptions.InvalidParameter,
        match=re.escape(
            "The parameter 'transmissivity' must be in the interval [0, 1]: transmissivity=-0.1"  # noqa: E501
        ),
    ):
        pq.simulate(program=program, number_of_modes=5)


def test_UniformLoss_transmissivity_invalid():
    with pq.Program() as program:
        pq.Q() | pq.NumberState([1, 1, 1, 0, 0])

        pq.Q() | pq.UniformLoss(0.5j)

    with pytest.raises(
        pq.api.exceptions.InvalidParameter,
        match=re.escape(
            "The parameter 'transmissivity' must be a single real number in the interval [0, 1]: transmissivity=0.5j"  # noqa: E501
        ),
    ):
        pq.simulate(program=program, number_of_modes=5)
