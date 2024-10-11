from piquasso.api.result import Result


def correlations(state, instruction, shots):
    state._correlations = instruction.params["correlations"]
    return Result(state=state)


def gaussian_hamiltonian(state, instruction, shots):
    H = instruction.params["hamiltonian"]

    # validate_fermionic_gaussian_hamiltonian(H)

    np = state._connector.np

    d = state.d

    A = H[d:, d:]
    B = H[:d, d:]

    A_plus_B = A + B
    A_minus_B = A - B

    h = np.block(
        [
            [A_plus_B.imag, A_plus_B.real],
            [-A_minus_B.real, A_minus_B.imag],
        ]
    )

    SO = state._connector.expm(-2 * h)

    # state._correlations[0] = SO @ state._correlations[0] @ SO.T
    state._correlations[0] = np.einsum("ik,jl,kl->ij", SO, SO, state._correlations[0])

    return Result(state=state)
