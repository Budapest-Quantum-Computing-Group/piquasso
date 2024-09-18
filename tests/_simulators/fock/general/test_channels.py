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


def test_Attenuator_with_zero_theta_changes_nothing_on_one_mode_state():
    with pq.Program() as empty_program:
        pq.Q() | pq.DensityMatrix(bra=[1], ket=[1]) / 2
        pq.Q() | pq.DensityMatrix(bra=[2], ket=[2]) / 2

    with pq.Program() as program_with_zero_theta:
        pq.Q() | pq.DensityMatrix(bra=[1], ket=[1]) / 2
        pq.Q() | pq.DensityMatrix(bra=[2], ket=[2]) / 2

        pq.Q() | pq.Attenuator(theta=0.0)

    simulator = pq.FockSimulator(d=1, config=pq.Config(cutoff=3))

    empty_state = simulator.execute(empty_program).state
    lossy_state = simulator.execute(program_with_zero_theta).state

    assert np.allclose(
        empty_state.fock_probabilities,
        lossy_state.fock_probabilities,
    )


def test_Attenuator_with_pi_over_2_theta_maps_to_vacuum():
    with pq.Program() as program_with_zero_theta:
        pq.Q() | pq.DensityMatrix(bra=[1], ket=[1]) / 2
        pq.Q() | pq.DensityMatrix(bra=[2], ket=[2]) / 2

        pq.Q() | pq.Attenuator(theta=np.pi / 2)

    simulator = pq.FockSimulator(d=1, config=pq.Config(cutoff=3))

    lossy_state = simulator.execute(program_with_zero_theta).state

    assert np.allclose(
        lossy_state.fock_probabilities,
        [1.0, 0.0, 0.0],
        atol=1e-7,
    )


def test_Attenuator_for_one_particle():
    theta = np.pi / 5

    transmittivity = np.cos(theta)

    with pq.Program() as program:
        pq.Q(0, 1) | pq.DensityMatrix(bra=[0, 1], ket=[0, 1])

        pq.Q(1) | pq.Attenuator(theta=theta)

    simulator = pq.FockSimulator(d=2, config=pq.Config(cutoff=2))

    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [1 - transmittivity**2, 0, transmittivity**2],
    )
