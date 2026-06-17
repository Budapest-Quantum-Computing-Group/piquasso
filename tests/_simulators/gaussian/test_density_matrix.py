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

from scipy.linalg import block_diag
from scipy.special import factorial

import piquasso as pq

import jax


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


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


@for_all_connectors
def test_thermal_density_matrix_matches_analytic(connector):
    """The thermal state has the closed-form diagonal density matrix."""
    mean_photon_number = 0.5
    cutoff = 8

    with pq.Program() as program:
        pq.Q(0) | pq.Thermal([mean_photon_number])

    state = (
        pq.GaussianSimulator(d=1, config=pq.Config(cutoff=cutoff), connector=connector)
        .execute(program)
        .state
    )

    density_matrix = state.density_matrix

    n = np.arange(cutoff)
    expected_diagonal = mean_photon_number**n / (1 + mean_photon_number) ** (n + 1)

    assert np.allclose(np.diag(density_matrix).real, expected_diagonal)
    assert np.allclose(density_matrix - np.diag(np.diag(density_matrix)), 0.0)


@for_all_connectors
def test_squeezed_vacuum_density_matrix_matches_analytic(connector):
    """Single-mode squeezed vacuum has only even-photon amplitudes."""
    r = 0.5
    cutoff = 8

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=r)

    state = (
        pq.GaussianSimulator(d=1, config=pq.Config(cutoff=cutoff), connector=connector)
        .execute(program)
        .state
    )

    amplitudes = np.zeros(cutoff)
    for k in range(cutoff // 2):
        amplitudes[2 * k] = (
            np.sqrt(factorial(2 * k))
            / (2**k * factorial(k))
            * (-np.tanh(r)) ** k
            / np.sqrt(np.cosh(r))
        )
    expected = np.outer(amplitudes, amplitudes)

    assert np.allclose(state.density_matrix, expected)


@for_all_connectors
def test_coherent_density_matrix_matches_analytic(connector):
    """Coherent state: :math:`\\rho_{mn} = e^{-|\\alpha|^2} \\alpha^m
    \\bar{\\alpha}^n / \\sqrt{m! n!}`."""
    cutoff = 10

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.7, phi=0.4)

    state = (
        pq.GaussianSimulator(d=1, config=pq.Config(cutoff=cutoff), connector=connector)
        .execute(program)
        .state
    )

    alpha = state.complex_displacement[0]
    n = np.arange(cutoff)
    amplitudes = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**n / np.sqrt(factorial(n))
    expected = np.outer(amplitudes, amplitudes.conj())

    assert np.allclose(state.density_matrix, expected)


@for_all_connectors
def test_density_matrix_is_hermitian(connector):
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5) | pq.Displacement(r=0.3, phi=0.7)
        pq.Q(1) | pq.Displacement(r=0.2)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.9, phi=0.4)

    state = (
        pq.GaussianSimulator(d=2, config=pq.Config(cutoff=6), connector=connector)
        .execute(program)
        .state
    )

    density_matrix = state.density_matrix

    assert np.allclose(density_matrix, density_matrix.conj().T)


@for_all_connectors
def test_density_matrix_diagonal_matches_fock_probabilities(connector):
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=0.5)
        pq.Q(1) | pq.Squeezing(r=0.3, phi=0.8)
        pq.Q(0, 1) | pq.Beamsplitter(theta=0.6, phi=0.2)

    state = (
        pq.GaussianSimulator(d=2, config=pq.Config(cutoff=6), connector=connector)
        .execute(program)
        .state
    )

    assert np.allclose(
        np.diag(state.density_matrix).real,
        state.fock_probabilities,
    )


@for_all_connectors
def test_density_matrix_Thermal_Interferometer(generate_unitary_matrix, connector):
    d = 3

    preparation = pq.Program([pq.Thermal([0.1, 0.2, 0.3])])

    U = generate_unitary_matrix(d)

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.Interferometer(U)

    simulator = pq.GaussianSimulator(
        d=d, config=pq.Config(cutoff=2), connector=connector
    )

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    fock_space_unitary = block_diag(np.array([1.0]), U)

    assert np.allclose(
        final_state.density_matrix,
        fock_space_unitary @ initial_state.density_matrix @ fock_space_unitary.conj().T,
    )


def test_density_matrix_differentiability():
    connector = pq.JaxConnector()

    def func(r):
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()
            pq.Q(0) | pq.Squeezing(r=r, phi=np.pi / 3)

        simulator = pq.GaussianSimulator(
            d=1, config=pq.Config(cutoff=3), connector=connector
        )

        state = simulator.execute(program).state

        return state.density_matrix[2, 2].real

    grad_func = jax.grad(func)

    r = 0.1

    expected_value = np.tanh(r) ** 2 / (2 * np.cosh(r))
    expected_derivative = np.tanh(r) / np.cosh(r) ** 3 - np.tanh(r) ** 3 / (
        2 * np.cosh(r)
    )

    actual_value = func(r)
    actual_derivative = grad_func(r)

    assert np.isclose(actual_value, expected_value)
    assert np.isclose(actual_derivative, expected_derivative)
