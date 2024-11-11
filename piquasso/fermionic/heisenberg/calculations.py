from piquasso.api.result import Result


from ..utils import (
    get_fermionic_orthogonal_transformation,
    validate_fermionic_gaussian_hamiltonian,
)

from .state import HeisenbergState


def correlations(state: HeisenbergState, instruction, shots):
    state._correlations = instruction.params["correlations"]
    return Result(state=state)

def interferometer(state: HeisenbergState, instruction, shots):
    unitary = instruction._get_passive_block(state._connector, state._config)
    modes = instruction.modes

    d = state._d

    connector = state._connector

    np = connector.np

    reU = np.real(unitary)
    imU = np.imag(unitary)

    SO = np.block([[reU, imU], [-imU, reU]])

    from piquasso._math.transformations import from_xxpp_to_xpxp_transformation_matrix

    T = from_xxpp_to_xpxp_transformation_matrix(state.d)


    SO = T @ SO @ T.T

    l = len(state._correlations.shape) // 2

    i = "abcdefghijk"[:2*l]
    j = "lmnopqrstuv"[:2*l]

    einsum_string = ""
    for k in range(2*l):
        einsum_string = einsum_string + f",{i[k]}{j[k]}"

    einsum_string = einsum_string[1:] + "," + j + "->" + i

    state._correlations = np.einsum(einsum_string, *([SO]*(2*l)), state._correlations)

    return Result(state=state)


def gaussian_hamiltonian(state, instruction, shots):
    H = instruction.params["hamiltonian"]

    validate_fermionic_gaussian_hamiltonian(H)

    np = state._connector.np

    SO = get_fermionic_orthogonal_transformation(H, state._connector)

    from piquasso._math.transformations import from_xxpp_to_xpxp_transformation_matrix

    T = from_xxpp_to_xpxp_transformation_matrix(state.d)

    SO = T @ SO @ T.T

    # state._correlations[0] = SO @ state._correlations[0] @ SO.T
    state._correlations = np.einsum("ik,jl,kl->ij", SO, SO, state._correlations)

    return Result(state=state)
