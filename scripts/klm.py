import piquasso as pq
import numpy as np

from scipy.stats import unitary_group

from piquasso._backends.fock.pure.calculations.passive_linear import _get_interferometer_on_fock_space, _apply_passive_gate_matrix_to_state
from piquasso._math.fock import cutoff_cardinality, get_fock_space_basis
from piquasso._math.indices import get_index_in_fock_space_array, get_index_in_fock_space

import os


from typing import Tuple


np.set_printoptions(suppress=True, linewidth=200)


def calculate_BS_matrix(
        d: int,
        modes: np.ndarray,
        params: np.ndarray,
        config: pq.Config,
        calculator
) -> np.ndarray:
    theta = params[0]
    phi = params[1]

    t = np.cos(theta)
    r = np.exp(1j * phi) * np.sin(theta)

    one_mode_matrix = np.array(
        [
            [t, -np.conj(r)],
            [r, t]
        ],
        dtype=config.complex_dtype
    )

    m = np.eye(d, dtype=config.complex_dtype)
    m[np.ix_(modes, modes)] = one_mode_matrix

    return m


def calculate_phase_shift_matrix(phis: np.ndarray, calculator) -> np.ndarray:
    np = calculator.np

    return np.diag(np.exp(1j * phis))


def calculate_matrix(
        d: int,
        config: pq.Config,
        bs_params: np.ndarray,
        phase_shift_params,
        calculator
) -> np.ndarray:
    phase_shifters = calculate_phase_shift_matrix(phis=phase_shift_params, calculator=calculator)

    bs1 = calculate_BS_matrix(
        d=d,
        modes=np.array([0, 1]),
        params=bs_params[0],
        config=config,
        calculator=calculator
    )

    bs2 = calculate_BS_matrix(
        d=d,
        modes=np.array([1, 2]),
        params=bs_params[1],
        config=config,
        calculator=calculator
    )

    bs3 = calculate_BS_matrix(
        d=d,
        modes=np.array([0, 1]),
        params=bs_params[2],
        config=config,
        calculator=calculator
    )

    return phase_shifters @ bs3 @ bs2 @ bs1


def get_lossy(U: np.ndarray, P: np.ndarray, k, calculator):
    np = calculator.np
    cutoff = P.shape[0]
    d = U.shape[0]
    overall_cutoff = cutoff + np.sum(k)
    card = cutoff_cardinality(cutoff=overall_cutoff, d=d)

    subspace_transformations = _get_interferometer_on_fock_space(
        U, overall_cutoff, calculator
    )

    range_ = np.arange(cutoff)
    k_repeated = np.repeat(k[np.newaxis, :], cutoff, axis=0)

    occupation_numbers = np.concatenate([range_[:, np.newaxis], k_repeated], axis=1)
    indices = get_index_in_fock_space_array(occupation_numbers)

    state_vector = np.zeros(shape=(card, cutoff), dtype=U.dtype)
    state_vector[indices, range_] = 1

    new_state = _apply_passive_gate_matrix_to_state(
        state_vector, subspace_transformations, d, overall_cutoff, (0, 1, 2), calculator
    )

    q_vectors = get_fock_space_basis(d=d-1, cutoff=cutoff)
    q_card = len(q_vectors)
    ps = np.zeros(q_card, dtype=P.dtype)

    for idx, q in enumerate(q_vectors):
        ps[idx] = P[1, q[0]] * P[0, q[1]]

    all_occupation_numbers = get_fock_space_basis(d=d, cutoff=overall_cutoff)

    result = np.zeros((cutoff, cutoff), dtype=U.dtype)
    for idx, n in enumerate(all_occupation_numbers):
        q = n[1:]
        ps_idx = get_index_in_fock_space(q)
        if n[0] < cutoff and np.sum(q) < cutoff:
            result[n[0]] += new_state[idx] * ps[ps_idx]

    breakpoint()




def main() -> None:
    cutoff = 3
    d = 3
    P = np.array([
        [1, 0.1, 0.2],
        [0, 0.9, 0.2],
        [0, 0, 0.6]
    ])
    P = np.eye(cutoff)
    k = np.array([1, 0])
    calculator = pq.NumpyCalculator()
    config = pq.Config(cutoff=cutoff)

    U = calculate_matrix(
        d=3,
        config=config,
        bs_params=np.array([
            [0.1, 0.2],  # first BS
            [0.2, 0.3],  # second BS
            [0.3, 0.3],  # third BS
        ]),
        phase_shift_params=np.array([0.1, 0.2, 0.3]),
        calculator=calculator
    )
    a = get_lossy(
        U=U,
        P=P,
        k=k,
        calculator=calculator
    )
    breakpoint()


if __name__ == "__main__":
    main()
