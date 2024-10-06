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


def test_FockState_reduced():
    with pq.Program() as program:
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

    simulator = pq.FockSimulator(d=2)
    state = simulator.execute(program).state

    with pq.Program() as reduced_program:
        pq.Q() | pq.DensityMatrix(ket=(1,), bra=(1,)) / 4

        pq.Q() | pq.DensityMatrix(ket=(2,), bra=(2,)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0,), bra=(0,)) / 2

    reduced_program_simulator = pq.FockSimulator(d=1)
    reduced_program_state = reduced_program_simulator.execute(reduced_program).state

    expected_reduced_state = reduced_program_state

    reduced_state = state.reduced(modes=(1,))

    assert expected_reduced_state == reduced_state


def test_FockState_fock_probabilities_map():
    with pq.Program() as program:
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

    simulator = pq.FockSimulator(d=2)
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


def test_FockState_quadratures_mean_variance():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.2, phi=0)
        pq.Q(1) | pq.Squeezing(r=0.1, phi=1.0)
        pq.Q(2) | pq.Displacement(r=0.2, phi=np.pi / 2)

    simulator = pq.FockSimulator(d=3, config=pq.Config(cutoff=8, hbar=2))
    result = simulator.execute(program)

    mean_on_0th, variance_on_0th = result.state.quadratures_mean_variance(modes=(0,))
    mean_on_1st, variance_on_1st = result.state.quadratures_mean_variance(modes=(1,))
    mean_on_2nd, variance_on_2nd = result.state.quadratures_mean_variance(
        modes=(0,), phi=np.pi / 2
    )
    mean_on_3rd, variance_on_3rd = result.state.quadratures_mean_variance(
        modes=(2,), phi=np.pi / 2
    )

    assert np.isclose(mean_on_0th, 0.4, rtol=1e-5)
    assert mean_on_1st == 0.0
    assert np.isclose(mean_on_2nd, 0.0)
    assert np.isclose(mean_on_3rd, 0.4, rtol=1e-5)
    assert np.isclose(variance_on_0th, 1.0, rtol=1e-5)
    assert np.isclose(variance_on_1st, 0.9112844, rtol=1e-5)
    assert np.isclose(variance_on_2nd, 1, rtol=1e-5)
    assert np.isclose(variance_on_3rd, 1, rtol=1e-5)


def test_FockState_non_selfadjoint_density_matrix_raises_InvalidState():

    state = pq.FockState(d=1, connector=pq.NumpyConnector())

    non_selfadjoint_matrix = np.array([[1, 2], [3, 4]])

    state._density_matrix = non_selfadjoint_matrix

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.validate()

    assert "The density matrix is not self-adjoint" in error.value.args[0]


def test_FockState_non_selfadjoint_density_matrix_no_error_if_validate_False():

    state = pq.FockState(
        d=1, connector=pq.NumpyConnector(), config=pq.Config(validate=False)
    )

    non_selfadjoint_matrix = np.array([[1, 2], [3, 4]])

    state._density_matrix = non_selfadjoint_matrix

    state.validate()
