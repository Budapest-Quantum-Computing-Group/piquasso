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
from piquasso._math.fock import nb_get_fock_space_basis
from piquasso._simulators.gaussian.probabilities import (
    DisplacedDensityMatrixCalculation,
    NondisplacedDensityMatrixCalculation,
)


@pytest.fixture
def connector():
    return pq.NumpyConnector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nondisplaced_calc(state, connector):
    return NondisplacedDensityMatrixCalculation(state.complex_covariance, connector)


def _displaced_calc(state, connector):
    return DisplacedDensityMatrixCalculation(
        state.complex_displacement, state.complex_covariance, connector
    )


def _hafnian_density_matrix(calc, basis):
    """Reference implementation: one hafnian call per element."""
    from scipy.special import factorial

    n = basis.shape[0]
    dm = np.empty((n, n), dtype=complex)
    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            reduce_on = np.concatenate([ket, bra])
            dm[i, j] = (
                calc._normalization
                * calc.calculate_hafnian(reduce_on)
                / np.sqrt(np.prod(factorial(reduce_on)))
            )
    return dm


# ---------------------------------------------------------------------------
# Nondisplaced states
# ---------------------------------------------------------------------------


def test_recurrence_vacuum_state(connector):
    """Vacuum state: rho[0,0] = 1, all other elements zero."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=5)
    dm = calc.get_density_matrix(basis)

    assert np.isclose(dm[0, 0], 1.0)
    assert np.allclose(dm[1:, :], 0.0)
    assert np.allclose(dm[:, 1:], 0.0)


def test_recurrence_squeezed_vacuum_analytic_values(connector):
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

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=8)
    dm = calc.get_density_matrix(basis)

    assert np.isclose(dm[0, 0].real, 1.0 / np.cosh(r), atol=1e-8)
    assert np.isclose(dm[2, 0].real, -np.tanh(r) / (np.cosh(r) * np.sqrt(2)), atol=1e-8)


def test_recurrence_squeezed_vacuum_odd_fock_states_zero(connector):
    """Squeezed vacuum has no population in odd Fock states."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=8)
    dm = calc.get_density_matrix(basis)

    assert np.allclose(dm[1::2, :], 0.0)
    assert np.allclose(dm[:, 1::2], 0.0)


def test_recurrence_nondisplaced_matches_hafnian_single_mode(connector):
    """Recurrence result matches element-wise hafnian for a 1-mode squeezed state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.4)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=6)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-10)


def test_recurrence_nondisplaced_matches_hafnian_two_mode(connector):
    """Recurrence result matches hafnian for a 2-mode entangled state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=0.0)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=2, cutoff=4)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-10)


def test_recurrence_nondisplaced_matches_hafnian_three_mode(connector):
    """Recurrence result matches hafnian for a 3-mode state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.4, phi=0.2)
        pq.Q(1, 2) | pq.Beamsplitter(theta=0.3, phi=0.1)

    sim = pq.GaussianSimulator(d=3, config=pq.Config(cutoff=3))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=3, cutoff=3)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-8)


def test_recurrence_nondisplaced_is_hermitian(connector):
    """Density matrix must be Hermitian."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.4, phi=0.3)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=2, cutoff=5)
    dm = calc.get_density_matrix(basis)

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_recurrence_nondisplaced_is_positive_semidefinite(connector):
    """All eigenvalues of the density matrix must be non-negative."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=0.0)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=2, cutoff=5)
    dm = calc.get_density_matrix(basis)

    eigenvalues = np.linalg.eigvalsh(dm)
    assert eigenvalues.min() >= -1e-10


def test_recurrence_nondisplaced_trace_at_most_one(connector):
    """Trace of the density matrix must be at most 1 (truncation can reduce it)."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.4)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=10))
    state = sim.execute(prog).state

    calc = _nondisplaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=10)
    dm = calc.get_density_matrix(basis)

    trace = np.trace(dm).real
    assert trace <= 1.0 + 1e-10
    assert trace > 0.9


# ---------------------------------------------------------------------------
# Displaced states
# ---------------------------------------------------------------------------


def test_recurrence_coherent_state_purity(connector):
    """A coherent state is pure, so tr(rho^2) should be close to 1."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=0.5)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=12))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=12)
    dm = calc.get_density_matrix(basis)

    purity = np.trace(dm @ dm).real
    assert np.isclose(purity, 1.0, atol=1e-3)


def test_recurrence_displaced_matches_hafnian_single_mode(connector):
    """Recurrence result matches hafnian for a 1-mode displaced squeezed state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.4)
        pq.Q(0) | pq.Displacement(r=0.5, phi=0.3)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=6))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=6)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-10)


def test_recurrence_displaced_matches_hafnian_two_mode(connector):
    """Recurrence result matches hafnian for a 2-mode displaced state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.4, phi=0.2)
        pq.Q(0) | pq.Displacement(r=0.4, phi=0.1)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=3))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=2, cutoff=3)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-10)


def test_recurrence_displaced_matches_hafnian_three_mode(connector):
    """Recurrence result matches hafnian for a 3-mode displaced state."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.3)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.4, phi=0.2)
        pq.Q(1, 2) | pq.Beamsplitter(theta=0.3, phi=0.1)
        pq.Q(0) | pq.Displacement(r=0.5, phi=0.7)

    sim = pq.GaussianSimulator(d=3, config=pq.Config(cutoff=3))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=3, cutoff=3)

    dm_rec = calc.get_density_matrix(basis)
    dm_ref = _hafnian_density_matrix(calc, basis)

    assert np.allclose(dm_rec, dm_ref, atol=1e-8)


def test_recurrence_displaced_is_hermitian(connector):
    """Density matrix of a displaced state must be Hermitian."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(0) | pq.Displacement(r=1.0, phi=0.7)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=8))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=8)
    dm = calc.get_density_matrix(basis)

    assert np.allclose(dm, dm.conj().T, atol=1e-10)


def test_recurrence_displaced_is_positive_semidefinite(connector):
    """All eigenvalues of a displaced state density matrix must be non-negative."""
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.4)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.3, phi=0.1)
        pq.Q(0) | pq.Displacement(r=0.6, phi=0.4)

    sim = pq.GaussianSimulator(d=2, config=pq.Config(cutoff=4))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=2, cutoff=4)
    dm = calc.get_density_matrix(basis)

    eigenvalues = np.linalg.eigvalsh(dm)
    assert eigenvalues.min() >= -1e-10


# ---------------------------------------------------------------------------
# Index ordering
# ---------------------------------------------------------------------------


def test_recurrence_index_ordering_matches_get_density_matrix_element(connector):
    """
    dm[i, j] from get_density_matrix must equal
    get_density_matrix_element(bra=basis[i], ket=basis[j]) for all i, j.
    This locks down the ket-oplus-bra index convention used in the recurrence.
    """
    with pq.Program() as prog:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.3)
        pq.Q(0) | pq.Displacement(r=0.4, phi=0.2)

    sim = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=5))
    state = sim.execute(prog).state

    calc = _displaced_calc(state, connector)
    basis = nb_get_fock_space_basis(d=1, cutoff=5)
    dm = calc.get_density_matrix(basis)

    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            expected = calc.get_density_matrix_element(bra=bra, ket=ket)
            assert np.isclose(dm[i, j], expected, atol=1e-10), (
                f"Mismatch at bra={bra}, ket={ket}: "
                f"got {dm[i, j]}, expected {expected}"
            )
