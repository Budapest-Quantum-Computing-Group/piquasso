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

from piquasso._math.fock import get_fock_space_basis


def _reference_density_matrix(state):
    """The original per-element (loop) hafnian evaluation, used as a reference."""
    calculation = state._get_density_matrix_calculation()

    basis = get_fock_space_basis(d=state.d, cutoff=state._config.cutoff)

    cardinality = len(basis)
    density_matrix = np.empty(shape=(cardinality, cardinality), dtype=complex)

    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            density_matrix[i, j] = calculation.get_density_matrix_element(bra, ket)

    return density_matrix


def _execute(program, d, cutoff):
    simulator = pq.GaussianSimulator(d=d, config=pq.Config(cutoff=cutoff))
    return simulator.execute(program).state


@pytest.mark.parametrize("cutoff", [4, 6, 8])
@pytest.mark.parametrize("displaced", [False, True])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_density_matrix_recurrence_matches_per_element_hafnian(d, displaced, cutoff):
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

    state = _execute(program, d, cutoff)

    density_matrix = state.density_matrix

    assert np.allclose(density_matrix, _reference_density_matrix(state))


def test_density_matrix_recurrence_for_mixed_state():
    """A lossy (mixed, non-displaced) state exercises the off-diagonal recurrence."""
    d = 2

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Squeezing(r=0.6)
        pq.Q(1) | pq.Squeezing(r=0.4, phi=0.5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.8, phi=0.2)
        pq.Q(0) | pq.Attenuator(theta=0.3)

    state = _execute(program, d, cutoff=7)

    density_matrix = state.density_matrix

    assert np.allclose(density_matrix, _reference_density_matrix(state))


def test_connector_without_accelerated_path_returns_none():
    """A connector that does not override the hook signals no accelerated path."""
    connector = pq.JaxConnector()

    assert connector.density_matrix_from_representation(None, None, None, None) is None


def test_density_matrix_falls_back_when_no_accelerated_path(monkeypatch):
    """Connectors without an accelerated path use the per-element evaluation."""
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5) | pq.Displacement(r=0.3)
        pq.Q(1) | pq.Squeezing(r=0.2, phi=0.4)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.6, phi=0.2)

    state = _execute(program, d=2, cutoff=6)

    accelerated = state.density_matrix

    monkeypatch.setattr(
        state._connector,
        "density_matrix_from_representation",
        lambda *args, **kwargs: None,
    )

    assert np.allclose(state.density_matrix, accelerated)


def test_density_matrix_is_hermitian():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5) | pq.Displacement(r=0.3, phi=0.7)
        pq.Q(1) | pq.Displacement(r=0.2)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.9, phi=0.4)

    state = _execute(program, d=2, cutoff=6)

    density_matrix = state.density_matrix

    assert np.allclose(density_matrix, density_matrix.conj().T)


def test_density_matrix_diagonal_matches_fock_probabilities():
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(1) | pq.Squeezing(r=0.3, phi=0.8)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.6, phi=0.2)

    state = _execute(program, d=2, cutoff=6)

    assert np.allclose(
        np.diag(state.density_matrix).real,
        state.fock_probabilities,
    )
