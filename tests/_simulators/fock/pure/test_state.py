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


def test_PureFockState_reduced():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=2)
    state = simulator.execute(program).state

    with pq.Program() as reduced_program:
        pq.Q() | pq.DensityMatrix(ket=(0,), bra=(0,)) / 2

        pq.Q() | pq.DensityMatrix(ket=(1,), bra=(1,)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2,), bra=(2,)) / 4

        pq.Q() | pq.DensityMatrix(ket=(1,), bra=(2,)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2,), bra=(1,)) / 4

    reduced_program_simulator = pq.FockSimulator(d=1)
    reduced_program_state = reduced_program_simulator.execute(reduced_program).state

    expected_reduced_state = reduced_program_state

    reduced_state = state.reduced(modes=(1,))

    assert expected_reduced_state == reduced_state


def test_PureFockState_reduced_preserves_Config():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=10))
    state = simulator.execute(program).state

    assert state._config.cutoff == 10


def test_PureFockState_fock_probabilities_map():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=2)
    state = simulator.execute(program).state

    expected_fock_probabilities = {
        (0, 0): 0.0,
        (0, 1): 0.25,
        (1, 0): 0.0,
        (0, 2): 0.25,
        (1, 1): 0.0,
        (2, 0): 0.5,
        (0, 3): 0.0,
        (1, 2): 0.0,
        (2, 1): 0.0,
        (3, 0): 0.0,
    }

    actual_fock_probabilities = state.fock_probabilities_map

    for occupation_number, expected_probability in expected_fock_probabilities.items():
        assert np.isclose(
            actual_fock_probabilities[occupation_number],
            expected_probability,
        )


def test_PureFockState_quadratures_mean_variance():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.2, phi=0)
        pq.Q(1) | pq.Squeezing(r=0.1, phi=1.0)
        pq.Q(2) | pq.Displacement(r=0.2, phi=np.pi / 2)

    simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=8, hbar=2))
    result = simulator.execute(program)

    mean_1, var_1 = result.state.quadratures_mean_variance(modes=(0,))
    mean_2, var_2 = result.state.quadratures_mean_variance(modes=(1,))
    mean_3, var_3 = result.state.quadratures_mean_variance(modes=(0,), phi=np.pi / 2)
    mean_4, var_4 = result.state.quadratures_mean_variance(modes=(2,), phi=np.pi / 2)

    assert np.isclose(mean_1, 0.4, rtol=0.00001)
    assert mean_2 == 0.0
    assert np.isclose(mean_3, 0.0)
    assert np.isclose(mean_4, 0.4, rtol=0.00001)
    assert np.isclose(var_1, 1.0, rtol=0.00001)
    assert np.isclose(var_2, 0.9112844, rtol=0.00001)
    assert np.isclose(var_3, 1, rtol=0.00001)
    assert np.isclose(var_4, 1, rtol=0.00001)


def test_PureFockState_mean_position():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.2, phi=0)

    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=8))
    state = simulator.execute(program).state

    assert np.isclose(state.mean_position(mode=0), 0.4)


def test_mean_position():
    d = 1
    cutoff = 7

    alpha_ = 0.02

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(r=alpha_)

    config = pq.Config(cutoff=cutoff)

    simulator = pq.PureFockSimulator(
        d=d, config=config, connector=pq.TensorflowConnector()
    )

    state = simulator.execute(program).state
    mean = state.mean_position(mode=0)

    assert np.allclose(mean, np.sqrt(2 * config.hbar) * alpha_)


def test_normalize_if_disabled_in_Config():
    d = 1
    cutoff = 3

    alpha_ = 1.0

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(all) | pq.Displacement(r=alpha_)

    config = pq.Config(cutoff=cutoff)

    simulator = pq.PureFockSimulator(d=d, config=config)

    state = simulator.execute(program).state
    norm = state.norm

    assert not np.isclose(norm, 1.0)


@pytest.mark.parametrize("connector", (pq.NumpyConnector(), pq.JaxConnector()))
def test_PureFockState_get_tensor_representation(connector):
    d = 2
    cutoff = 3

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff), connector=connector
    )
    state = simulator.execute(program).state

    state_tensor = state.get_tensor_representation()

    assert state_tensor.shape == (3,) * 2

    assert np.allclose(
        state_tensor,
        np.array(
            [
                [0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
    )


def test_PureFockState_get_purity():
    d = 2
    cutoff = 3

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1]) / 2

        pq.Q() | pq.StateVector([0, 2]) / 2
        pq.Q() | pq.StateVector([2, 0]) / np.sqrt(2)

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))
    state = simulator.execute(program).state

    purity = state.get_purity()

    assert np.isclose(purity, 1.0)


@pytest.mark.monkey
def test_PureFockState_get_particle_detection_probability_on_modes():
    d = 2
    cutoff = 3

    state_vector_length = 6

    coeffs = np.random.uniform(size=state_vector_length) + 1j * np.random.uniform(
        size=state_vector_length
    )

    coeffs /= np.linalg.norm(coeffs)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0]) * coeffs[0]
        pq.Q() | pq.StateVector([0, 1]) * coeffs[1]
        pq.Q() | pq.StateVector([1, 0]) * coeffs[2]
        pq.Q() | pq.StateVector([0, 2]) * coeffs[3]
        pq.Q() | pq.StateVector([1, 1]) * coeffs[4]
        pq.Q() | pq.StateVector([2, 0]) * coeffs[5]

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))
    state = simulator.execute(program).state

    state.validate()

    probability_of_0_on_mode_1 = state.get_particle_detection_probability_on_modes(
        occupation_numbers=(0,), modes=(1,)
    )

    assert np.isclose(
        probability_of_0_on_mode_1,
        np.abs(coeffs[0]) ** 2 + np.abs(coeffs[2]) ** 2 + np.abs(coeffs[5]) ** 2,
    )


def test_PureFockState_displaced_squeezed_state_mean_photon_number():
    d = 2
    cutoff = 10

    r_S = 0.2
    r_D = 0.3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r_S, phi=np.pi / 5)
        pq.Q(0) | pq.Displacement(r_D, phi=np.pi / 3)

        pq.Q(0, 1) | pq.Beamsplitter()

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))
    state = simulator.execute(program).state

    mean = state.mean_photon_number()

    expected_mean = np.sinh(r_S) ** 2 + r_D**2

    assert np.isclose(mean, expected_mean)


def test_PureFockState_displaced_state_variance_photon_number():
    d = 2
    cutoff = 10

    r = 0.3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r, np.pi / 5)

        pq.Q(0, 1) | pq.Beamsplitter()

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))
    state = simulator.execute(program).state

    variance = state.variance_photon_number()

    assert np.isclose(variance, r**2)


def test_PureFockState_unnormalized_state_raises_InvalidState():
    d = 1
    low_cutoff = 3

    high_displacement = 3.0

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(high_displacement)

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=low_cutoff))
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.validate()

    assert "The sum of probabilities is" in error.value.args[0]


def test_PureFockState_unnormalized_state_validate_False():
    d = 1
    low_cutoff = 3

    high_displacement = 3.0

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.Displacement(high_displacement)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=low_cutoff, validate=False)
    )
    state = simulator.execute(program).state

    assert not np.isclose(state.norm, 1.0)

    state.validate()
