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

from piquasso.api.exceptions import InvalidParameter, InvalidState


def test_Mean_is_automatically_scaled_by_hbar():
    config = pq.Config(hbar=42)

    xpxp_mean_vector = np.array([1, 2])

    with pq.Program() as program:
        pq.Q() | pq.Mean(xpxp_mean_vector)

    simulator = pq.GaussianSimulator(d=1, config=config)

    result = simulator.execute(program)

    assert np.allclose(
        result.state.xpxp_mean_vector, xpxp_mean_vector * np.sqrt(config.hbar)
    )


def test_Covariance_is_automatically_scaled_by_hbar():
    config = pq.Config(hbar=42)

    xpxp_covariance_matrix = np.array(
        [
            [2, 1],
            [1, 2],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(xpxp_covariance_matrix)

    simulator = pq.GaussianSimulator(d=1, config=config)

    result = simulator.execute(program)

    assert np.allclose(
        result.state.xpxp_covariance_matrix, xpxp_covariance_matrix * config.hbar
    )


def test_Thermal_is_automatically_scaled_by_hbar():
    config = pq.Config(hbar=42)

    mean_photon_numbers = np.array([1, 2])

    with pq.Program() as program:
        pq.Q() | pq.Thermal(mean_photon_numbers)

    simulator = pq.GaussianSimulator(d=2, config=config)

    result = simulator.execute(program)

    assert np.allclose(
        result.state.xpxp_covariance_matrix,
        config.hbar
        * np.array(
            [
                [3, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 5, 0],
                [0, 0, 0, 5],
            ]
        ),
    )


def test_Thermal_with_zero_mean_photon_numbers_yields_Vacuum():
    config = pq.Config(hbar=42)

    mean_photon_numbers = np.array([0, 0])

    with pq.Program() as thermal_program:
        pq.Q() | pq.Thermal(mean_photon_numbers)

    with pq.Program() as vacuum_program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=2, config=config)

    thermal_state = simulator.execute(thermal_program).state
    vacuum_state = simulator.execute(vacuum_program).state

    assert thermal_state == vacuum_state


def test_state_initialization_with_misshaped_mean():
    misshaped_mean = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Mean(misshaped_mean)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_misshaped_covariance():
    misshaped_cov = np.array(
        [
            [1, 2, 10000],
            [1, 1, 10000],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(misshaped_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_nonsymmetric_covariance():
    nonsymmetric_cov = np.array(
        [
            [1, 2],
            [1, 1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(nonsymmetric_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_state_initialization_with_nonpositive_covariance():
    nonpositive_cov = np.array(
        [
            [1, 0],
            [0, -1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Covariance(nonpositive_cov)

    simulator = pq.GaussianSimulator(d=1)

    with pytest.raises(InvalidState):
        simulator.execute(program)


def test_Thermal_with_negative_mean_photon_numbers_raises_InvalidParameter():
    mean_photon_numbers = np.array([-1, 1])

    with pytest.raises(InvalidParameter):
        pq.Q() | pq.Thermal(mean_photon_numbers)


def test_vacuum_resets_the_state(state):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.GaussianSimulator(d=state.d)

    new_state = simulator.execute(program, initial_state=state).state

    assert np.allclose(
        new_state.xpxp_mean_vector,
        np.zeros(2 * new_state.d),
    )
    assert np.allclose(
        new_state.xpxp_covariance_matrix,
        np.identity(2 * new_state.d) * simulator.config.hbar,
    )
