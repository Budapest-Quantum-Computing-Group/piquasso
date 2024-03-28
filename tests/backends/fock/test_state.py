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

import pytest

import numpy as np

import piquasso as pq


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_get_particle_detection_probability(SimulatorClass):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.1, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    simulator = SimulatorClass(d=2, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state
    state.normalize()

    probability = state.get_particle_detection_probability(occupation_number=(0, 2))

    assert np.isclose(probability, 0.0012355767142126952)


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_quadratures_mean_variance(SimulatorClass):
    alpha = 0.2 - 0.2j

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=6))
    state = simulator.execute(program).state
    state.normalize()

    mean, var = state.quadratures_mean_variance(modes=(0,))

    assert np.isclose(mean, 0.3275267, atol=1e-5)
    assert np.isclose(var, 0.6709301, atol=1e-4)


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_wigner_function(SimulatorClass):
    alpha = 1 - 0.5j

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state
    state.normalize()

    wigner_function_values = state.wigner_function(
        positions=[1, 1.1],
        momentums=[-0.5, -0.6],
    )

    assert np.allclose(
        wigner_function_values,
        np.array([[0.09872782, 0.10777084], [0.10327266, 0.11273187]]),
    )


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_wigner_function_raises_InvalidModes_for_multiple_modes(
    SimulatorClass,
):
    alpha = 1 - 0.5j

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = SimulatorClass(d=2)
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidModes):
        state.wigner_function(
            positions=[1, 1.1],
            momentums=[-0.5, -0.6],
        )


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_wigner_function_raises_InvalidModes_for_multiple_modes_specified(
    SimulatorClass,
):
    alpha = 1 - 0.5j

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(0) | pq.Squeezing(r=0.2)

    simulator = SimulatorClass(d=2)
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidModes):
        state.wigner_function(
            positions=[1, 1.1],
            momentums=[-0.5, -0.6],
        )


@pytest.mark.parametrize("SimulatorClass", (pq.FockSimulator, pq.PureFockSimulator))
def test_FockState_fidelity(
    SimulatorClass,
):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.2, phi=np.pi / 2)

    simulator = SimulatorClass(d=1, config=pq.Config(cutoff=6))
    state = simulator.execute(program).state
    state.normalize()

    with pq.Program() as program_2:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.2, phi=-np.pi / 2)

    state_2 = simulator.execute(program_2).state
    state_2.normalize()
    fid = state.fidelity(state_2)

    assert np.isclose(fid, 0.92507584)
    assert np.isclose(state_2.fidelity(state), fid)
    assert np.isclose(state_2.fidelity(state_2), 1.0)


def test_FockState_get_purity():
    with pq.Program() as program:
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 1, 1)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 2)) * np.sqrt(1 / 8)

    simulator = pq.FockSimulator(d=3, config=pq.Config(cutoff=3))

    state = simulator.execute(program).state

    state.validate()

    purity = state.get_purity()

    assert np.isclose(purity, 0.625)
