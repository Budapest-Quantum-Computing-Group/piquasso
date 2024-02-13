#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

"""
Some of the code has been copyied from
`https://strawberryfields.ai/photonics/demos/run_gate_synthesis.html`.
"""

import pytest

import numpy as np

import piquasso as pq
from piquasso import cvqnn

from functools import partial

from jax import grad, jit


@pytest.fixture
def cutoff():
    return 10


@pytest.fixture
def layer_count():
    return 3


@pytest.fixture
def d():
    return 2


def piquasso_benchmark(benchmark, d, cutoff, layer_count):
    weights = cvqnn.generate_random_cvqnn_weights(layer_count, d)

    benchmark(lambda : _calculate_piquasso_results(weights, cutoff))


@partial(jit, static_argnums=1)
def _calculate_mean_position(weights, cutoff):
    d = cvqnn.get_number_of_modes(weights.shape[1])

    simulator = pq.PureFockSimulator(
        d=d,
        config=pq.Config(cutoff=cutoff, normalize=False),
        calculator=pq.JaxCalculator(),
    )

    program = cvqnn.create_program(weights)

    state = simulator.execute(program).state

    return state.mean_position(0)


_mean_position_grad = partial(jit, static_argnums=1)(
    grad(_calculate_mean_position, argnums=0)
)


def _calculate_piquasso_results(weights, cutoff):
    state_vector = _calculate_mean_position(weights, cutoff)

    return state_vector, _mean_position_grad(weights, cutoff)
