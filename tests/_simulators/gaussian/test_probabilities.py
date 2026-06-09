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

import numpy as np
import pytest

import piquasso as pq


# ---------------------------------------------------------------------------
# Nondisplaced states
# ---------------------------------------------------------------------------


def test_density_matrix_vacuum_state():
    """Vacuum state: rho[0,0] = 1, all other elements zero."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.isclose(dm[0, 0], 1.0)
    assert np.allclose(dm[1:, :], 0.0)
    assert np.allclose(dm[:, 1:], 0.0)


def test_density_matrix_squeezed_vacuum_analytic_values():
    """
    For a single-mode squeezed vacuum with parameter r the analytic values are:

        rho[0,0] = 1 / cosh(r)
        rho[2,0] = -tanh(r) / (cosh(r) * sqrt(2))
    """
    r = 0.5

    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=r)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state
    dm = state.density_matrix

    # Fock basis ordering: |0>, |1>, |2>, ...  index 0 -> n=0, index 2 -> n=2
    assert np.isclose(dm[0, 0], 1.0 / np.cosh(r), atol=1e-10)
    assert np.isclose(
        dm[2, 0], -np.tanh(r) / (np.cosh(r) * np.sqrt(2)), atol=1e-10
    )


def test_density_matrix_squeezed_vacuum_odd_fock_states_zero():
    """Squeezed vacuum has no odd-photon-number contributions."""
    r = 0.5

    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=r)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state
    dm = state.density_matrix

    # Odd indices: 1, 3, 5, 7
    assert np.allclose(dm[1::2, :], 0.0, atol=1e-10)
    assert np.allclose(dm[:, 1::2], 0.0, atol=1e-10)


def test_density_matrix_nondisplaced_is_hermitian_single_mode():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.4)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_density_matrix_nondisplaced_is_hermitian_two_mode():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.5, phi=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.4, phi=0.1)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_density_matrix_nondisplaced_is_positive_semidefinite():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.3, phi=0.0)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=7))
    state = sim.execute(prog).state
    dm = state.density_matrix
    eigvals = np.linalg.eigvalsh(dm)

    assert np.all(eigvals >= -1e-10)


def test_density_matrix_nondisplaced_unit_trace():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.4, phi=0.0)
        pq.Q(1, 2) | pq.Beamsplitter(theta=0.3, phi=0.0)

    sim = pq.GaussianSimulator(d=3, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.isclose(np.trace(dm).real, 1.0, atol=1e-2)


def test_density_matrix_coherent_state_is_pure():
    """A coherent state is pure so Tr(rho^2) = 1."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=0.0)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=12))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.isclose(np.trace(dm @ dm).real, 1.0, atol=1e-2)


# ---------------------------------------------------------------------------
# Displaced states
# ---------------------------------------------------------------------------


def test_density_matrix_displaced_is_hermitian_single_mode():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.5, phi=0.3)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_density_matrix_displaced_is_hermitian_two_mode():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.3, phi=0.0)
        pq.Q(0) | pq.Displacement(r=0.5, phi=0.2)
        pq.Q(1) | pq.Displacement(r=0.3, phi=1.0)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_density_matrix_displaced_is_positive_semidefinite():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.3)
        pq.Q(0) | pq.Displacement(r=0.4, phi=0.5)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state
    dm = state.density_matrix
    eigvals = np.linalg.eigvalsh(dm)

    assert np.all(eigvals >= -1e-10)


def test_density_matrix_displaced_unit_trace():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.4, phi=0.5)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.isclose(np.trace(dm).real, 1.0, atol=1e-6)


def test_density_matrix_three_mode_displaced_is_hermitian():
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.3, phi=0.0)
        pq.Q(0, 2) | pq.Beamsplitter(theta=0.4, phi=0.1)
        pq.Q(0) | pq.Displacement(r=0.2, phi=0.0)
        pq.Q(2) | pq.Displacement(r=0.3, phi=1.0)

    sim = pq.GaussianSimulator(d=3, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state
    dm = state.density_matrix

    assert np.allclose(dm, dm.conj().T, atol=1e-10)