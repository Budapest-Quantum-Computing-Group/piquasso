import piquasso as pq
import numpy as np


def test_GaussianHamiltonian():
    d = 3

    A = np.array([[1, 2j, 3j], [-2j, 5, 6], [-3j, 6, 7]])
    B = np.array([[0, 1j, 2], [-1j, 0, 3], [-2, -3, 0]])

    hamiltonian = np.block([[-A.conj(), B], [-B.conj(), A]])

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=hamiltonian)

    simulator = pq.fermionic.GaussianSimulator(d=d)

    state = simulator.execute(program).state
    covariance_matrix = state.covariance_matrix
    second_order_correlations = (covariance_matrix + 1j * np.identity(2 * d)) / 2j

    with pq.Program() as program:
        pq.Q() | pq.fermionic.Correlations([second_order_correlations])
        pq.Q() | pq.fermionic.GaussianHamiltonian(hamiltonian=-hamiltonian)

    simulator = pq.fermionic.HeisenbergSimulator(d=d)
    state = simulator.execute(program).state
    ident = np.identity(d)

    assert np.allclose(
        state._correlations[0],
        np.block([[ident / 2, -1j * ident / 2], [1j * ident / 2, ident / 2]]),
    )
