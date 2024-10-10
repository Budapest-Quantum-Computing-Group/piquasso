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

import piquasso as pq

import numpy as np

from jax import jit, grad, jacrev


def test_Beamsplitter_mean_particle_numbers_grad():
    connector = pq.JaxConnector()

    def get_mean_particle_number_first_mode(theta):
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 0])

            pq.Q() | pq.Beamsplitter(theta=theta)

        simulator = pq.fermionic.GaussianSimulator(
            d=2, connector=connector, config=pq.Config(validate=False)
        )

        state = simulator.execute(program).state

        return state.mean_particle_numbers(modes=(0,))[0]

    jitted_func = jit(get_mean_particle_number_first_mode)

    theta = np.pi / 3
    result = jitted_func(theta)

    assert np.isclose(result, np.cos(theta) ** 2)

    grad_func = grad(jitted_func)
    result_grad = grad_func(theta)

    assert np.isclose(result_grad, -2 * np.cos(theta) * np.sin(theta))


def test_Beamsplitter_mean_particle_numbers_jacrev():
    connector = pq.JaxConnector()

    def get_mean_particle_numbers(theta):
        with pq.Program() as program:
            pq.Q() | pq.StateVector([1, 0])

            pq.Q() | pq.Beamsplitter(theta=theta)

        simulator = pq.fermionic.GaussianSimulator(
            d=2, connector=connector, config=pq.Config(validate=False)
        )

        state = simulator.execute(program).state

        return state.mean_particle_numbers(modes=(0, 1))

    jitted_func = jit(get_mean_particle_numbers)

    theta = np.pi / 3
    result = jitted_func(theta)

    assert np.isclose(result[0], np.cos(theta) ** 2)
    assert np.isclose(result[1], np.sin(theta) ** 2)

    jac_func = jacrev(jitted_func)
    result_jac = jac_func(theta)

    assert np.isclose(result_jac[0], -2 * np.cos(theta) * np.sin(theta))
    assert np.isclose(result_jac[1], 2 * np.cos(theta) * np.sin(theta))
