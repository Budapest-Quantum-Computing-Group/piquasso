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


def test_density_matrix_displaced_coherent_analytic_values():
    r"""Analytic check for a coherent state |alpha> with alpha = r*exp(i*phi).

    Piquasso stores ``dm[n, m] = alpha^n * conj(alpha)^m * exp(-|alpha|^2) / sqrt(n! m!)``,
    so the **row** index carries ``alpha`` (not its conjugate).
    """
    r_disp, phi = 0.5, 0.3
    alpha = r_disp * np.exp(1j * phi)

    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r_disp, phi=phi)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state
    dm = state.density_matrix

    from scipy.special import factorial

    for n in range(4):
        for m in range(4):
            expected = (
                np.exp(-(abs(alpha) ** 2))
                * alpha ** n
                * alpha.conj() ** m
                / np.sqrt(float(factorial(n)) * float(factorial(m)))
            )
            assert np.isclose(dm[n, m], expected, atol=1e-10), (
                f"dm[{n},{m}] = {dm[n,m]:.6f} != {expected:.6f}"
            )


def test_density_matrix_displaced_off_diagonal_phases():
    """Off-diagonal elements must carry the correct complex phase.

    For a coherent state |alpha> with non-zero phase phi:

        dm[0, 1] = alpha^0 * conj(alpha)^1 * exp(-|alpha|^2) = exp(-|alpha|^2) * conj(alpha)
        dm[1, 0] = alpha^1 * conj(alpha)^0 * exp(-|alpha|^2) = exp(-|alpha|^2) * alpha

    These are complex conjugates.  A ket/bra swap in the recurrence extraction
    would reverse their imaginary parts, making the test fail.
    """
    r_disp, phi = 0.5, 0.7  # non-zero phi makes the phase visible
    alpha = r_disp * np.exp(1j * phi)

    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=r_disp, phi=phi)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state
    dm = state.density_matrix

    # row=0, col=1: a^0 * conj(a)^1 = conj(a)
    expected_01 = np.exp(-(abs(alpha) ** 2)) * alpha.conj()
    # row=1, col=0: a^1 * conj(a)^0 = a
    expected_10 = np.exp(-(abs(alpha) ** 2)) * alpha

    assert np.isclose(dm[0, 1], expected_01, atol=1e-10)
    assert np.isclose(dm[1, 0], expected_10, atol=1e-10)
    # Sanity: they are conjugates of each other but differ in imaginary part.
    assert not np.isclose(dm[0, 1], dm[1, 0], atol=1e-10)


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


def test_density_matrix_recurrence_matches_elementwise_nondisplaced():
    """Recurrence result must exactly match the element-wise hafnian path."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.4, phi=0.2)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.3, phi=0.1)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state

    dm_recurrence = state.density_matrix

    # Force the element-wise hafnian path by calling the internal method directly.
    from piquasso._math.fock import get_fock_space_basis

    occ = get_fock_space_basis(d=2, cutoff=5)
    calc = state._get_density_matrix_calculation()
    dm_elementwise = calc._get_density_matrix_elementwise(occ)

    assert np.allclose(dm_recurrence, dm_elementwise, atol=1e-10)


def test_density_matrix_recurrence_matches_elementwise_displaced():
    """Recurrence result must match the element-wise path for displaced states."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0, 1) | pq.Squeezing2(r=0.3, phi=0.1)
        pq.Q(0) | pq.Displacement(r=0.4, phi=0.6)
        pq.Q(1) | pq.Displacement(r=0.2, phi=1.2)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state

    dm_recurrence = state.density_matrix

    from piquasso._math.fock import get_fock_space_basis

    occ = get_fock_space_basis(d=2, cutoff=4)
    calc = state._get_density_matrix_calculation()
    dm_elementwise = calc._get_density_matrix_elementwise(occ)

    assert np.allclose(dm_recurrence, dm_elementwise, atol=1e-10)