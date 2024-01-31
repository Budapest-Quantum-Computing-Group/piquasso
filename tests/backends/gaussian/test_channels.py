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


def test_DeterministicGaussianChannel_for_one_mode():
    X = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    Y = np.array(
        [
            [1.5, 0.0],
            [0.0, 1.5],
        ]
    )

    with pq.Program() as program:
        pq.Q(0) | pq.Squeezing(1.0)
        pq.Q(1) | pq.Squeezing(1.0)

        pq.Q(1) | pq.DeterministicGaussianChannel(X=X, Y=Y)

    simulator = pq.GaussianSimulator(d=2)

    state = simulator.execute(program).state

    state.validate()


def test_DeterministicGaussianChannel_for_multiple_modes():
    X = np.array(
        [
            [0.53069278, 0.95401384, 0.07446993, 0.42721325],
            [0.95401384, 0.12611099, 0.63393981, 1.59568682],
            [0.07446993, 0.63393981, 0.29636723, 1.30277118],
            [0.42721325, 1.59568682, 1.30277118, 1.44820453],
        ],
    )
    Y = np.array(
        [
            [1.78993994, 0.99247326, 0.74008171, 0.87280485],
            [0.99247326, 1.70466087, 1.12406614, 0.25481437],
            [0.74008171, 1.12406614, 1.9950749, 0.72310148],
            [0.87280485, 0.25481437, 0.72310148, 1.72001795],
        ]
    )

    with pq.Program() as program:
        pq.Q(0) | pq.Squeezing(1.0)
        pq.Q(1) | pq.Squeezing(1.0)

        pq.Q(0, 1) | pq.DeterministicGaussianChannel(X=X, Y=Y)

    simulator = pq.GaussianSimulator(d=2)

    state = simulator.execute(program).state
    state.validate()


def test_DeterministicGaussianChannel_raises_InvalidParameter_for_invalid_X():
    invalid_X = np.array(
        [
            [0.5j, 0.0],
            [0.0, 0.5],
            [1.0, 2.0],
        ]
    )

    Y = np.array(
        [
            [1.5, 0.0],
            [0.0, 1.5],
        ]
    )

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.DeterministicGaussianChannel(X=invalid_X, Y=Y)

    assert "The parameter 'X' must be a real 2n-by-2n matrix:" in error.value.args[0]


def test_DeterministicGaussianChannel_raises_InvalidParameter_for_invalid_Y():
    X = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    invalid_Y = np.array(
        [
            [0.5j, 0.0],
            [0.0, 0.5],
            [1.0, 2.0],
        ]
    )

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.DeterministicGaussianChannel(X=X, Y=invalid_Y)

    assert "The parameter 'Y' must be a real 2n-by-2n matrix:" in error.value.args[0]


@pytest.mark.monkey
def test_DeterministicGaussianChannel_raises_InvalidParameter_for_incompatible_shapes(
    generate_symmetric_matrix,
):
    X = generate_symmetric_matrix(2)
    Y = generate_symmetric_matrix(4)

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.DeterministicGaussianChannel(X=X, Y=Y)

    assert "The shape of matrices 'X' and 'Y' should be equal:" in error.value.args[0]


def test_DeterministicGaussianChannel_raises_InvalidParameter_for_invalid_X_and_Y():
    X = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    Y = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.DeterministicGaussianChannel(X=X, Y=Y)

    assert (
        "The matrices 'X' and 'Y' does not satisfy the inequality corresponding to "
        "Gaussian channels."
    ) in error.value.args[0]


def test_DeterministicGaussianChannel_raises_InvalidInstruction_at_incompatible_modes():
    X = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )

    Y = np.array(
        [
            [1.5, 0.0],
            [0.0, 1.5],
        ]
    )

    with pq.Program() as program:
        pq.Q(0, 1) | pq.DeterministicGaussianChannel(X=X, Y=Y)

    simulator = pq.GaussianSimulator(d=2)

    with pytest.raises(pq.api.exceptions.InvalidInstruction) as error:
        simulator.execute(program)

    assert "The instruction should be specified for '2' modes:" in error.value.args[0]


def test_Attenuator_with_zero_thermal_exciation():
    original_mean_photon_number = 2
    theta = np.pi / 6
    mean_thermal_excitation = 0

    with pq.Program() as program:
        for i in [0, 1]:
            pq.Q(i) | pq.Squeezing(np.arcsinh(np.sqrt(original_mean_photon_number)))

    with pq.Program() as attenuated_program:
        pq.Q(0, 1) | program

        pq.Q(0) | pq.Attenuator(
            theta=theta, mean_thermal_excitation=mean_thermal_excitation
        )

    simulator = pq.GaussianSimulator(d=2)

    state = simulator.execute(program).state
    mean_photon_number = state.mean_photon_number((0,))

    attenuated_state = simulator.execute(attenuated_program).state
    attenuated_mean_photon_number = attenuated_state.mean_photon_number((0,))

    assert np.isclose(mean_photon_number, original_mean_photon_number)
    assert np.isclose(
        attenuated_mean_photon_number, mean_photon_number * np.cos(theta) ** 2
    )


def test_Attenuator_with_multiple_modes():
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Squeezing2(0.3)

        pq.Q(0) | pq.Attenuator(theta=0.1, mean_thermal_excitation=0)

    simulator = pq.GaussianSimulator(d=2)

    state = simulator.execute(program).state

    assert np.isclose(state.mean_photon_number(modes=(0, 1)), 0.18454097911952028)


def test_Attenuator_with_nonzero_thermal_exciation():
    original_mean_photon_number = 2
    theta = np.pi / 6
    mean_thermal_excitation = 0.57721

    with pq.Program() as program:
        for i in [0, 1]:
            pq.Q(i) | pq.Squeezing(np.arcsinh(np.sqrt(original_mean_photon_number)))

    with pq.Program() as attenuated_program:
        pq.Q(0, 1) | program

        pq.Q(0) | pq.Attenuator(
            theta=theta, mean_thermal_excitation=mean_thermal_excitation
        )

    simulator = pq.GaussianSimulator(d=2)

    state = simulator.execute(program).state
    mean_photon_number = state.mean_photon_number((0,))

    attenuated_state = simulator.execute(attenuated_program).state
    attenuated_mean_photon_number = attenuated_state.mean_photon_number((0,))

    assert np.isclose(mean_photon_number, original_mean_photon_number)
    assert np.isclose(
        attenuated_mean_photon_number,
        (
            mean_photon_number
            * np.cos(theta) ** 2
            * (1 + mean_thermal_excitation * theta / np.pi)
        ),
    )


def test_Attenuator_raises_InvalidParameter_for_negative_thermal_exciations():
    theta = np.pi / 6
    invalid_mean_thermal_excitation = -1

    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        pq.Attenuator(
            theta=theta, mean_thermal_excitation=invalid_mean_thermal_excitation
        )

    assert (
        "The parameter 'mean_thermal_excitation' must be a positive real number:"
        in error.value.args[0]
    )


def test_Attenuator_raises_InvalidInstruction_for_multiple_modes():
    theta = np.pi / 6
    mean_thermal_excitation = 5

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Attenuator(
            theta=theta, mean_thermal_excitation=mean_thermal_excitation
        )

    simulator = pq.GaussianSimulator(d=2)

    with pytest.raises(pq.api.exceptions.InvalidInstruction) as error:
        simulator.execute(program)

    assert "The instruction should be specified for '2' modes:" in error.value.args[0]
