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

from functools import reduce

import numpy as np

import piquasso as pq

import pytest

from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric
from piquasso._math.validations import all_in_interval

from piquasso.fermionic.gaussian._misc import get_omega


for_all_connectors = pytest.mark.parametrize(
    "connector", [pq.NumpyConnector(), pq.JaxConnector()]
)


@for_all_connectors
def test_vacuum_covariance_matrix(connector):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    ident = np.identity(d)
    zeros = np.zeros_like(ident)

    state.validate()

    assert type(state) is pq.fermionic.GaussianState

    assert np.allclose(
        state.covariance_matrix, np.block([[zeros, ident], [-ident, zeros]])
    )


@for_all_connectors
def test_vacuum_correlation_matrix(connector):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    ident = np.identity(d)
    zeros = np.zeros_like(ident)

    assert np.allclose(
        state.correlation_matrix, np.block([[zeros, zeros], [zeros, ident]])
    )


@pytest.mark.monkey
@for_all_connectors
def test_vacuum_evolved_with_GaussianHamiltonian_is_valid(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 2

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    state.validate()


@pytest.mark.monkey
@for_all_connectors
def test_vacuum_evolved_with_GaussianHamiltonian_correlation_matrix_random(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    gamma = state.correlation_matrix

    gamma11 = gamma[:d, :d]
    gamma12 = gamma[:d, d:]
    gamma21 = gamma[d:, :d]
    gamma22 = gamma[d:, d:]

    assert is_selfadjoint(gamma11)
    assert is_selfadjoint(gamma22)

    assert is_skew_symmetric(gamma12)
    assert is_skew_symmetric(gamma21)

    assert np.allclose(gamma11, np.identity(d) - gamma22.conj())
    assert np.allclose(gamma12, -gamma21.conj())

    assert np.allclose(
        gamma @ gamma, gamma
    ), "The correlation matrix is pure, must be projector"
    assert np.isclose(
        np.trace(gamma), d
    ), "The correlation matrix must be a projector of rank 3"


@pytest.mark.monkey
@for_all_connectors
def test_GaussianHamiltonian_covariance_and_correlation_matrix_equivalence(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    omega = get_omega(d, connector)

    assert np.allclose(
        state.covariance_matrix,
        -1j
        * omega
        @ (2 * state.correlation_matrix - np.identity(2 * d))
        @ omega.conj().T,
    ), "Eq. (45) from https://arxiv.org/pdf/2111.08343 should hold"


@for_all_connectors
def test_vacuum_correlation_matrix_density_matrix_equivalence(
    connector,
    get_ladder_operators,
):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    fs = get_ladder_operators(d)

    rho = state.density_matrix

    correlation_matrix = state.correlation_matrix

    expected_correlation_matrix = np.empty_like(correlation_matrix)

    for i in range(2 * d):
        for j in range(2 * d):
            expected_correlation_matrix[i, j] = np.trace(rho @ fs[i].T @ fs[j])

    assert np.allclose(correlation_matrix, expected_correlation_matrix)


@pytest.mark.monkey
@for_all_connectors
def test_correlation_matrix_density_matrix_equivalence_random(
    connector,
    generate_fermionic_gaussian_hamiltonian,
    get_ladder_operators,
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    fs = get_ladder_operators(d)

    rho = state.density_matrix

    correlation_matrix = state.correlation_matrix

    expected_correlation_matrix = np.empty_like(correlation_matrix)

    for i in range(2 * d):
        for j in range(2 * d):
            expected_correlation_matrix[i, j] = np.trace(rho @ fs[i].T @ fs[j])

    assert np.allclose(correlation_matrix, expected_correlation_matrix)


@for_all_connectors
def test_vacuum_covariance_matrix_with_majorana_operators_and_density_matrix(
    connector, get_majorana_operators
):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    m = get_majorana_operators(d)

    covariance_matrix = state.covariance_matrix

    rho = state.density_matrix

    for i in range(2 * d):
        for j in range(2 * d):
            assert np.isclose(
                covariance_matrix[i, j],
                -1j * np.trace(rho @ (m[i] @ m[j] - m[j] @ m[i])) / 2,
            )


@pytest.mark.monkey
@for_all_connectors
def test_covariance_matrix_with_majorana_operators_and_density_matrix(
    connector, generate_fermionic_gaussian_hamiltonian, get_majorana_operators
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    m = get_majorana_operators(d)

    covariance_matrix = state.covariance_matrix

    rho = state.density_matrix

    expected_covariance_matrix = np.empty(shape=(2 * d, 2 * d), dtype=rho.dtype)

    for i in range(2 * d):
        for j in range(2 * d):
            expected_covariance_matrix[i, j] = (
                -1j * np.trace(rho @ (m[i] @ m[j] - m[j] @ m[i])) / 2
            )

    assert np.allclose(covariance_matrix, expected_covariance_matrix)


@pytest.mark.monkey
@for_all_connectors
def test_GaussianHamiltonian_correlation_matrix_and_maj_equivalence(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    omega = get_omega(d, connector)

    assert np.allclose(
        state.maj_correlation_matrix,
        omega @ state.correlation_matrix @ omega.conj().T,
    ), "Eq. (43) from https://arxiv.org/pdf/2111.08343 should hold"


@for_all_connectors
def test_vacuum_evolved_with_GaussianHamiltonian_correlation_matrix(connector):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    assert np.allclose(
        state.correlation_matrix,
        np.array(
            [
                [
                    0.33416157 - 0.0j,
                    -0.0401874 + 0.01146957j,
                    -0.05663529 - 0.00376422j,
                    0.0 + 0.0j,
                    -0.19852528 + 0.31881949j,
                    0.05833792 - 0.27031065j,
                ],
                [
                    -0.0401874 - 0.01146957j,
                    0.23155699 - 0.0j,
                    -0.15017863 - 0.05386455j,
                    0.19852528 - 0.31881949j,
                    0.0 + 0.0j,
                    -0.04634951 + 0.08677748j,
                ],
                [
                    -0.05663529 + 0.00376422j,
                    -0.15017863 + 0.05386455j,
                    0.13234052 - 0.0j,
                    -0.05833792 + 0.27031065j,
                    0.04634951 - 0.08677748j,
                    0.0 - 0.0j,
                ],
                [
                    -0.0 + 0.0j,
                    0.19852528 + 0.31881949j,
                    -0.05833792 - 0.27031065j,
                    0.66583843 - 0.0j,
                    0.0401874 + 0.01146957j,
                    0.05663529 - 0.00376422j,
                ],
                [
                    -0.19852528 - 0.31881949j,
                    -0.0 + 0.0j,
                    0.04634951 + 0.08677748j,
                    0.0401874 - 0.01146957j,
                    0.76844301 - 0.0j,
                    0.15017863 - 0.05386455j,
                ],
                [
                    0.05833792 + 0.27031065j,
                    -0.04634951 - 0.08677748j,
                    -0.0 - 0.0j,
                    0.05663529 + 0.00376422j,
                    0.15017863 + 0.05386455j,
                    0.86765948 - 0.0j,
                ],
            ]
        ),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_2_terms_consecutive(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 1]),
        np.trace(density_matrix @ m[0] @ m[1]),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_2_terms(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([1, 3]),
        np.trace(density_matrix @ m[1] @ m[3]),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_4_terms(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 1, 3, 4]),
        np.trace(density_matrix @ m[0] @ m[1] @ m[3] @ m[4]),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_4_terms_consecutive(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 1, 2, 3]),
        np.trace(density_matrix @ m[0] @ m[1] @ m[2] @ m[3]),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_4_terms_flipped(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 2, 1, 3]),
        np.trace(density_matrix @ m[0] @ m[2] @ m[1] @ m[3]),
    )

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 1, 2, 3]),
        -state.get_majorana_monomial_expectation_value([0, 2, 1, 3]),
    ), "Flips should change the sign."


@for_all_connectors
def test_get_majorana_monomial_expectation_value_with_duplicates(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([0, 2, 0, 1, 2, 3]),
        np.trace(density_matrix @ m[0] @ m[2] @ m[0] @ m[1] @ m[2] @ m[3]),
    )


@for_all_connectors
def test_get_majorana_monomial_expectation_value_with_triple(
    connector, get_majorana_operators
):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value([5, 5, 0, 3, 0, 0, 1, 4]),
        np.trace(
            density_matrix @ m[5] @ m[5] @ m[0] @ m[3] @ m[0] @ m[0] @ m[1] @ m[4]
        ),
    )


@pytest.mark.monkey
@for_all_connectors
def test_get_majorana_monomial_expectation_value_random_without_multiplicities(
    connector, get_majorana_operators, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    initial_length = np.random.randint(1, 10)

    indices = np.random.randint(0, 2 * d, initial_length)

    indices = list(set(indices))  # Remove duplicates

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    majorana_string = reduce(np.dot, [m[i] for i in indices])

    assert np.isclose(
        state.get_majorana_monomial_expectation_value(indices),
        np.trace(density_matrix @ majorana_string),
    )


@pytest.mark.monkey
@for_all_connectors
def test_get_majorana_monomial_expectation_value_empty(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state

    density_matrix = state.density_matrix

    empty_indices = np.array([], dtype=int)

    assert np.isclose(
        state.get_majorana_monomial_expectation_value(empty_indices),
        np.trace(density_matrix),
    )


@pytest.mark.monkey
@for_all_connectors
def test_get_majorana_monomial_expectation_value_random_with_possible_multiplicities(
    connector, get_majorana_operators, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    initial_length = np.random.randint(1, 10)

    indices = np.random.randint(0, 2 * d, initial_length)

    density_matrix = state.density_matrix

    m = get_majorana_operators(d)

    majorana_string = reduce(np.dot, [m[i] for i in indices])

    assert np.isclose(
        state.get_majorana_monomial_expectation_value(indices),
        np.trace(density_matrix @ majorana_string),
    )


@pytest.mark.monkey
@for_all_connectors
def test_reduced_state_correlation_matrix_random(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    reduced_state = state.reduced(modes=(0, 1))

    gamma = reduced_state.correlation_matrix
    dr = d - 1

    gamma11 = gamma[:dr, :dr]
    gamma12 = gamma[:dr, dr:]
    gamma21 = gamma[dr:, :dr]
    gamma22 = gamma[dr:, dr:]

    assert is_selfadjoint(gamma11)
    assert is_selfadjoint(gamma22)

    assert is_skew_symmetric(gamma12)
    assert is_skew_symmetric(gamma21)

    assert np.allclose(gamma11, np.identity(dr) - gamma22.conj())
    assert np.allclose(gamma12, -gamma21.conj())


@for_all_connectors
def test_reduced_state_correlation_matrix(connector):
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    reduced_state = state.reduced(modes=(0, 1))
    reduced_state.validate()

    assert np.allclose(
        reduced_state.correlation_matrix,
        np.array(
            [
                [
                    0.33416157 - 0.0j,
                    -0.0401874 + 0.01146957j,
                    0.0 + 0.0j,
                    -0.19852528 + 0.31881949j,
                ],
                [
                    -0.0401874 - 0.01146957j,
                    0.23155699 - 0.0j,
                    0.19852528 - 0.31881949j,
                    0.0 + 0.0j,
                ],
                [
                    -0.0 + 0.0j,
                    0.19852528 + 0.31881949j,
                    0.66583843 - 0.0j,
                    0.0401874 + 0.01146957j,
                ],
                [
                    -0.19852528 - 0.31881949j,
                    -0.0 + 0.0j,
                    0.0401874 - 0.01146957j,
                    0.76844301 - 0.0j,
                ],
            ]
        ),
    )


@pytest.mark.monkey
@for_all_connectors
def test_vacuum_evolved_with_passive_GaussianHamiltonian_stays_vacuum(
    connector,
    generate_passive_fermionic_gaussian_hamiltonian,
):
    d = 3

    hamiltonian = generate_passive_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    with pq.Program() as vacuum_program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state
    vacuum_state = simulator.execute(vacuum_program).state

    assert state == vacuum_state


@pytest.mark.monkey
@for_all_connectors
def test_correlation_matrix_ParentHamiltonian(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    assert np.allclose(
        state.correlation_matrix,
        np.linalg.inv(np.identity(2 * d) + connector.expm(2 * parent_hamiltonian)),
    )


@for_all_connectors
def test_get_parent_hamiltonian_Vacuum(connector):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.PiquassoException) as error:
        state.get_parent_hamiltonian()

    error_message = error.value.args[0]

    assert error_message == (
        "Cannot calculate parent Hamiltonian, since the correlation matrix is "
        "singular."
    )


@pytest.mark.monkey
@for_all_connectors
def test_parent_hamiltonian_roundtrip(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    assert np.allclose(parent_hamiltonian, state.get_parent_hamiltonian())


@for_all_connectors
def test_ParentHamiltonian_invalid_hamiltonian_raises_InvalidState(connector):
    d = 1

    parent_hamiltonian = np.array([[1, 2], [3, 4]])

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    with pytest.raises(pq.api.exceptions.InvalidParameter) as error:
        simulator.execute(program).state

    error_message = error.value.args[0]

    assert error_message == "Invalid Hamiltonian specified."


@for_all_connectors
def test_density_matrix_single_mode(connector):
    eps = 1 / 3

    parent_hamiltonian = np.array([[eps, 0], [0, -eps]])

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=1, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    assert is_selfadjoint(density_matrix)
    eigvals = np.linalg.eigvals(density_matrix)

    assert all_in_interval(eigvals, 0, 1)

    assert np.isclose(np.trace(density_matrix), 1.0)

    f = 1 / (1 + np.exp(2 * eps))

    assert np.allclose(density_matrix, np.array([[1 - f, 0], [0, f]]))


@for_all_connectors
def test_density_matrix_two_mode_simple(connector):
    eps1 = 1 / 3
    eps2 = 1 / 4

    eps = np.array([eps1, eps2])

    parent_hamiltonian = np.diag(np.concatenate([eps, -eps]))

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=2, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    assert is_selfadjoint(density_matrix)
    eigvals = np.linalg.eigvals(density_matrix)

    assert all_in_interval(eigvals, 0, 1)

    assert np.isclose(np.trace(density_matrix), 1.0)

    f = 1 / (1 + np.exp(2 * eps))

    from piquasso.fermionic.gaussian._misc import tensor_product

    single_mode_dms = []

    for fi in f:
        single_mode_dms.append(np.diag([1 - fi, fi]))

    assert np.allclose(density_matrix, tensor_product(single_mode_dms))


@for_all_connectors
@pytest.mark.monkey
def test_density_matrix_2_mode(connector, generate_fermionic_gaussian_hamiltonian):
    d = 2

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    assert is_selfadjoint(density_matrix)
    eigvals = np.linalg.eigvals(density_matrix)

    assert all_in_interval(eigvals, 0, 1)

    assert np.isclose(np.trace(density_matrix), 1.0)

    bigH = pq.fermionic.gaussian._misc.get_fermionic_hamiltonian(
        parent_hamiltonian, connector
    )

    assert np.allclose(
        density_matrix, connector.expm(bigH) / np.trace(connector.expm(bigH))
    )


@pytest.mark.monkey
@for_all_connectors
def test_density_matrix(connector, generate_fermionic_gaussian_hamiltonian):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    density_matrix = state.density_matrix

    assert is_selfadjoint(density_matrix)
    eigvals = np.linalg.eigvals(density_matrix)

    assert all_in_interval(eigvals, 0, 1)

    assert np.isclose(np.trace(density_matrix), 1.0)


@for_all_connectors
def test_identical_state_vector_overlap(connector):
    d = 3

    state_vector = [1, 0, 1]

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    assert np.isclose(state.overlap(state), 1.0)


@for_all_connectors
def test_different_state_vector_overlap(connector):
    d = 3

    state_vector_1 = [1, 0, 1]

    state_vector_2 = [0, 1, 1]

    with pq.Program() as program_1:
        pq.Q() | pq.StateVector(state_vector_1)

    with pq.Program() as program_2:
        pq.Q() | pq.StateVector(state_vector_2)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state_1 = simulator.execute(program_1).state
    state_2 = simulator.execute(program_2).state

    assert np.isclose(state_1.overlap(state_2), 0.0)
    assert np.isclose(state_2.overlap(state_1), 0.0)


@pytest.mark.monkey
@for_all_connectors
def test_covariance_matrix_GaussianHamiltonian_equivalence_from_Vacuum(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    omega = get_omega(d, connector)

    gate_hamiltonian_majorana = -1j * omega @ gate_hamiltonian @ omega.conj().T

    covariance_unitary = connector.expm(-2 * gate_hamiltonian_majorana)

    assert np.allclose(
        final_state.covariance_matrix,
        covariance_unitary
        @ initial_state.covariance_matrix
        @ covariance_unitary.conj().T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_correlation_matrix_GaussianHamiltonian_equivalence_from_Vacuum(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    correlation_unitary = connector.expm(2j * gate_hamiltonian)

    assert np.allclose(
        final_state.correlation_matrix,
        correlation_unitary
        @ initial_state.correlation_matrix
        @ correlation_unitary.conj().T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_density_matrix_GaussianHamiltonian_equivalence_from_Vacuum(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as preparation:
        pq.Q() | pq.Vacuum()

    with pq.Program() as program:
        pq.Q() | preparation
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    initial_state = simulator.execute(preparation).state
    final_state = simulator.execute(program).state

    bigH = pq.fermionic.gaussian._misc.get_fermionic_hamiltonian(
        gate_hamiltonian, connector
    )

    gate_unitary = connector.expm(1j * bigH)

    assert np.allclose(
        final_state.density_matrix,
        gate_unitary @ initial_state.density_matrix @ gate_unitary.conj().T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_density_matrix_GaussianHamiltonian_equivalence_1_mode(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 1

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)
    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    initial_state = simulator.execute(program).state

    with pq.Program() as program:
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    final_state = simulator.execute(program, initial_state=initial_state).state

    bigH = pq.fermionic.gaussian._misc.get_fermionic_hamiltonian(
        gate_hamiltonian, connector
    )

    gate_unitary = connector.expm(1j * bigH)

    assert np.allclose(
        final_state.density_matrix,
        gate_unitary @ initial_state.density_matrix @ gate_unitary.conj().T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_density_matrix_GaussianHamiltonian_equivalence_2_mode(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 2

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)
    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    initial_state = simulator.execute(program).state

    with pq.Program() as program:
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    final_state = simulator.execute(program, initial_state=initial_state).state

    bigH = pq.fermionic.gaussian._misc.get_fermionic_hamiltonian(
        gate_hamiltonian, connector
    )

    gate_unitary = connector.expm(1j * bigH)

    assert np.allclose(
        final_state.density_matrix,
        gate_unitary @ initial_state.density_matrix @ gate_unitary.conj().T,
    )


@pytest.mark.monkey
@for_all_connectors
def test_density_matrix_GaussianHamiltonian_equivalence(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3

    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)
    gate_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    initial_state = simulator.execute(program).state

    with pq.Program() as program:
        pq.Q() | pq.fermionic.GaussianHamiltonian(gate_hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    final_state = simulator.execute(program, initial_state=initial_state).state

    bigH = pq.fermionic.gaussian._misc.get_fermionic_hamiltonian(
        gate_hamiltonian, connector
    )

    gate_unitary = connector.expm(1j * bigH)

    assert np.allclose(
        final_state.density_matrix,
        gate_unitary @ initial_state.density_matrix @ gate_unitary.conj().T,
    )


@for_all_connectors
def test_Vacuum_density_matrix(connector):
    d = 2

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

    simulator = pq.fermionic.GaussianSimulator(d=d)

    state = simulator.execute(program).state

    assert np.allclose(state.density_matrix, np.diag([1, 0, 0, 0]))


@for_all_connectors
def test_StateVector_density_matrix(connector):
    d = 2

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 0])

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    assert np.allclose(state.density_matrix, np.diag([0, 0, 1, 0]))


@for_all_connectors
def test_parity_StateVector(connector):
    d = 3

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1, 0])

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    parity = state.get_parity_operator_expectation_value()

    assert np.isclose(parity, -1)


@pytest.mark.monkey
@for_all_connectors
def test_parity_StateVector_random(connector):
    d = 3

    state_vector = np.random.randint(0, 2, d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector(state_vector)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    parity = state.get_parity_operator_expectation_value()

    assert np.isclose(parity, (-1) ** np.sum(state_vector))


@pytest.mark.monkey
@for_all_connectors
def test_parity_is_invariant_under_linear_transformations(
    connector,
    generate_fermionic_gaussian_hamiltonian,
):
    d = 3

    H = generate_fermionic_gaussian_hamiltonian(d)

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 1, 0])

        pq.Q() | pq.fermionic.GaussianHamiltonian(H)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state = simulator.execute(program).state

    parity = state.get_parity_operator_expectation_value()

    assert np.isclose(parity, -1)


@pytest.mark.monkey
@for_all_connectors
def test_overlap_Interferometer(connector, generate_unitary_matrix):
    d = 3

    U1 = generate_unitary_matrix(d)
    U2 = generate_unitary_matrix(d)

    state_vector = [1, 0, 1]

    with pq.Program() as program1:
        pq.Q() | pq.StateVector(state_vector)
        pq.Q() | pq.Interferometer(U1)

    with pq.Program() as program2:
        pq.Q() | pq.StateVector(state_vector)
        pq.Q() | pq.Interferometer(U2)

    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)

    state1 = simulator.execute(program1).state
    state2 = simulator.execute(program2).state

    state1.validate()
    state2.validate()

    expected_overlap = np.trace(state1.density_matrix @ state2.density_matrix)

    assert np.isclose(state1.overlap(state2), expected_overlap)


@pytest.mark.monkey
@for_all_connectors
def test_overlap(connector, generate_fermionic_gaussian_hamiltonian):
    d = 3

    parent_hamiltonian_1 = generate_fermionic_gaussian_hamiltonian(d)
    parent_hamiltonian_2 = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian_1)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state1 = simulator.execute(program).state

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian_2)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state2 = simulator.execute(program).state

    state1.validate()
    state2.validate()

    expected_overlap = np.trace(state1.density_matrix @ state2.density_matrix)

    assert np.isclose(state1.overlap(state2), expected_overlap)


@pytest.mark.monkey
@for_all_connectors
def test_fock_probabilities_density_matrix_equivalence(
    connector, generate_fermionic_gaussian_hamiltonian
):
    d = 3
    parent_hamiltonian = generate_fermionic_gaussian_hamiltonian(d)

    program = pq.Program(
        [pq.fermionic.ParentHamiltonian(hamiltonian=parent_hamiltonian)]
    )
    simulator = pq.fermionic.GaussianSimulator(d=d, connector=connector)
    state = simulator.execute(program).state
    state.validate()

    fock_probabilities_from_density_matrix = np.real(np.diag(state.density_matrix))

    assert np.allclose(state.fock_probabilities, fock_probabilities_from_density_matrix)
