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

import numpy as np
import piquasso as pq

from jax import jit, jacfwd


def test_jit_compilation():
    def func(r, theta, xi):
        simulator = pq.PureFockSimulator(
            d=2,
            config=pq.Config(cutoff=5, dtype=np.float32),
            connector=pq.JaxConnector(),
        )

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

            pq.Q(0) | pq.Kerr(xi=xi)

        return simulator.execute(program).state.fock_probabilities

    r = 0.1
    theta = np.pi / 5
    xi = 0.01

    compiled_func = jit(func)

    result = func(r, theta, xi)
    compiled_result = compiled_func(r, theta, xi)

    assert np.allclose(result, compiled_result)


def test_jit_and_jacfwd_compilation():
    def func(r, theta, xi):
        simulator = pq.PureFockSimulator(
            d=2,
            config=pq.Config(cutoff=5, dtype=np.float32),
            connector=pq.JaxConnector(),
        )

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(r=r)

            pq.Q(0, 1) | pq.Beamsplitter(theta=theta)

            pq.Q(0) | pq.Kerr(xi=xi)

        return simulator.execute(program).state.fock_probabilities

    r = 0.1
    theta = np.pi / 5
    xi = 0.01

    jacobian = jacfwd(func, argnums=(0, 1, 2))

    compiled_jacobian = jit(jacobian)

    result = jacobian(r, theta, xi)
    compiled_result = compiled_jacobian(r, theta, xi)

    assert np.allclose(result[0], compiled_result[0])
    assert np.allclose(result[1], compiled_result[1])
    assert np.allclose(result[2], compiled_result[2])
