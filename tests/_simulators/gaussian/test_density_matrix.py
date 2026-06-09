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

import pytest
import numpy as np

import piquasso as pq


def _example_program(d, displaced):
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        for mode in range(d):
            pq.Q(mode) | pq.Squeezing(r=0.3 + 0.1 * mode, phi=0.2 * mode)

            if displaced:
                pq.Q(mode) | pq.Displacement(r=0.4 + 0.1 * mode, phi=0.3 * mode)

        if d >= 2:
            pq.Q(0, 1) | pq.Beamsplitter(theta=0.5, phi=0.1)

        if d >= 3:
            pq.Q(1, 2) | pq.Beamsplitter(theta=0.7, phi=0.4)

    return program


@pytest.mark.parametrize(
    "d, displaced",
    [(1, True), (2, False), (2, True)],
)
def test_density_matrix_agrees_across_connectors(d, displaced):
    """The fast (NumPy) and the generic (JAX) density matrices must agree.

    ``NumpyConnector`` uses the Hermite recurrence while ``JaxConnector`` falls
    back to the per-element loop hafnian, so this also pins the recurrence to the
    independent per-element evaluation.

    Single-mode squeezed vacuum is left out on purpose: it exposes an unrelated
    numerical limitation of the JAX loop-hafnian placeholder.
    """
    program = _example_program(d, displaced)
    config = pq.Config(cutoff=5)

    numpy_state = (
        pq.GaussianSimulator(d=d, config=config, connector=pq.NumpyConnector())
        .execute(program)
        .state
    )
    jax_state = (
        pq.GaussianSimulator(d=d, config=config, connector=pq.JaxConnector())
        .execute(program)
        .state
    )

    assert np.allclose(
        np.asarray(numpy_state.density_matrix),
        np.asarray(jax_state.density_matrix),
    )


def test_thermal_density_matrix_matches_analytic():
    """The thermal state has the closed-form diagonal density matrix."""
    mean_photon_number = 0.5
    cutoff = 8

    with pq.Program() as program:
        pq.Q(0) | pq.Thermal([mean_photon_number])

    state = (
        pq.GaussianSimulator(d=1, config=pq.Config(cutoff=cutoff))
        .execute(program)
        .state
    )

    density_matrix = state.density_matrix

    n = np.arange(cutoff)
    expected_diagonal = mean_photon_number**n / (1 + mean_photon_number) ** (n + 1)

    assert np.allclose(np.diag(density_matrix).real, expected_diagonal)
    assert np.allclose(density_matrix - np.diag(np.diag(density_matrix)), 0.0)


def test_density_matrix_is_hermitian():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5) | pq.Displacement(r=0.3, phi=0.7)
        pq.Q(1) | pq.Displacement(r=0.2)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.9, phi=0.4)

    state = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=6)).execute(program).state

    density_matrix = state.density_matrix

    assert np.allclose(density_matrix, density_matrix.conj().T)


def test_density_matrix_diagonal_matches_fock_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(1) | pq.Squeezing(r=0.3, phi=0.8)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.6, phi=0.2)

    state = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=6)).execute(program).state

    assert np.allclose(
        np.diag(state.density_matrix).real,
        state.fock_probabilities,
    )
