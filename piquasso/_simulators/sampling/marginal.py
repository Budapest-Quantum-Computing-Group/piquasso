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

r"""Exact marginal probabilities for ideal Fock-state Boson Sampling.

This module computes marginal output probabilities for a selected subset of
modes, optionally together with a fixed postselection pattern. The implemented
method is based on the low-rank marginal generating function.

Let ``T`` denote the union of the postselected modes and the marginal modes,
and let ``k = len(T)``. For each occupied input mode ``a`` with occupation
``r_a``, define

    V[a, j] = conjugate(U[T[j], a])

and

    f_a(z) = sum_j V[a, j] z_j.

The central polynomial is

    P(z, w) =
        prod_a L_{r_a}(-f_a(z) conjugate(f_a(w))),

where ``L_r`` is the Laguerre polynomial. If

    P(z, w) = sum_{alpha, beta} P_{alpha, beta} z^alpha w^beta,

then the scaled diagonal coefficients

    b_alpha = alpha! P_{alpha, alpha}

are the coefficients of the low-rank permanent polynomial

    G(c) =
        1 / r! * Per((I + V diag(c) V^dagger)_{r,r}).

The probability of observing a pattern ``y`` on the selected modes is obtained
from the binomial transform

    Pr(Y = y) =
        sum_{alpha >= y, |alpha| <= n}
            b_alpha
            prod_j binom(alpha_j, y_j) (-1)^(alpha_j - y_j).

The implementation computes the coefficients ``b_alpha`` by multiplying the
Laguerre factors in a bihomogeneous block representation. Since every factor
has equal total degree in the ``z`` variables and in the ``w`` variables, all
intermediate products satisfy

    P_{alpha, beta} = 0 whenever |alpha| != |beta|.

Thus the coefficient table decomposes into blocks

    P_d = (P_{alpha, beta})_{|alpha| = |beta| = d},

which avoids storing the full rectangular coefficient array.

If postselected modes are supplied, the returned values are joint probabilities
with the postselection event. Conditional postselected probabilities can be
obtained by normalizing the returned values by their sum.

For ``k`` selected modes and ``n`` photons, the block storage size is

    sum_{d=0}^n binom(d + k - 1, k - 1)^2.

For collision-free input and fixed ``k``, the sequential dynamic program has
time scaling O(n^(2k)) and memory scaling O(n^(2k-1)).

NOTE: This implementation might be improved by a sparse polynomial multiplication
algorithm, which could perhaps reduce the time and memory requirements of this method;
see https://arxiv.org/abs/0901.4323.
"""

from typing import Dict, Tuple, Sequence

import numpy as np
import numba as nb

from scipy.special import factorial

from piquasso._math.combinatorics import comb
from piquasso._math.fock import get_fock_space_basis, cutoff_fock_space_dim_array
from piquasso._math.indices import get_index_in_fock_subspace


def get_marginal_probabilities(
    input_photons: np.ndarray,
    interferometer: np.ndarray,
    postselected_modes: Tuple[int, ...],
    postselected_photons: Tuple[int, ...],
    marginal_modes: Tuple[int, ...],
) -> Dict[Tuple[int, ...], float]:
    r"""Calculates the marginal probabilities in a Boson Sampling experiment.

    Args:
        input_photons: The input photon numbers in each mode.
        interferometer: The unitary matrix describing the interferometer.
        postselected_modes: The modes on which a postselection is already performed.
        postselected_photons: The photon numbers corresponding to the postselection.
        marginal_modes: The modes for which the marginal probabilities are calculated.

    Returns:
        Dict[Tuple[int, ...], float]: The marginal probabilities of the state.
    """
    all_modes = np.asarray(postselected_modes + marginal_modes, dtype=np.int64)
    k_total = len(all_modes)
    k_marginal = len(marginal_modes)

    n = np.sum(input_photons)
    n_post = np.sum(postselected_photons)

    compositions = get_fock_space_basis(k_total, n + 1)

    diagonal_coefficients = _get_diagonal_coefficients(
        input_photons=input_photons,
        interferometer=interferometer,
        all_modes=all_modes,
        compositions=compositions,
    )

    outcomes = get_fock_space_basis(k_marginal, n - n_post + 1)

    num_postselected_modes = len(postselected_modes)

    outcomes_with_postselection = np.zeros((len(outcomes), k_total), dtype=np.int64)
    if num_postselected_modes:
        outcomes_with_postselection[:, :num_postselected_modes] = np.asarray(
            postselected_photons, dtype=np.int64
        )
    outcomes_with_postselection[:, num_postselected_modes:] = outcomes
    probabilities = _probabilities_from_diagonal_coefficients(
        diagonal_coefficients, compositions, outcomes_with_postselection
    )

    ret = {tuple(outcome): probabilities[i].real for i, outcome in enumerate(outcomes)}

    return ret


def _get_diagonal_coefficients(
    input_photons: np.ndarray,
    interferometer: np.ndarray,
    all_modes: np.ndarray,
    compositions: np.ndarray,
) -> np.ndarray:
    n = np.sum(input_photons)
    k = len(all_modes)

    occupied_input_modes = np.nonzero(input_photons)[0]
    occupations = input_photons[occupied_input_modes]

    V = interferometer[np.ix_(all_modes, occupied_input_modes)].T.conj()

    indices = cutoff_fock_space_dim_array(np.arange(n + 2), k)

    bihomogeneous_blocks = nb.typed.List(
        [
            np.zeros((indices[i + 1] - indices[i],) * 2, dtype=interferometer.dtype)
            for i in range(n + 1)
        ]
    )
    bihomogeneous_blocks[0][0, 0] = 1.0

    factorials = factorial(np.arange(n + 1))

    processed_degree = 0

    for a, occupation in enumerate(occupations):
        new_blocks = nb.typed.List(
            [np.zeros_like(bihomogeneous_blocks[i]) for i in range(n + 1)]
        )

        coeffs = _linear_form_power_coefficients(
            row=V[a],
            compositions=compositions[: indices[occupation + 1]],
            factorials=factorials,
        )

        for i in range(processed_degree + 1):
            source_block = bihomogeneous_blocks[i]
            source_comps = compositions[indices[i] : indices[i + 1]]

            for j in range(occupation + 1):
                add_comps = compositions[indices[j] : indices[j + 1]]
                add_block = coeffs[indices[j] : indices[j + 1]]

                _add_bihomogeneous_product_term(
                    source_block,
                    add_block,
                    source_comps,
                    add_comps,
                    weight=comb(occupation, j) / factorials[j],
                    out_block=new_blocks[i + j],
                )

        bihomogeneous_blocks = new_blocks
        processed_degree = processed_degree + occupation

    return _extract_diagonal_coefficients_from_bihomogeneous_blocks(
        bihomogeneous_blocks, compositions, indices, factorials
    )


@nb.njit(cache=True)
def _extract_diagonal_coefficients_from_bihomogeneous_blocks(
    bihomogeneous_blocks: Sequence[np.ndarray],
    compositions: np.ndarray,
    indices: np.ndarray,
    factorials: np.ndarray,
) -> np.ndarray:
    n = len(factorials) - 1

    diagonal_coefficients = np.zeros(len(compositions), dtype=np.float64)

    offsets = np.zeros(n + 2, dtype=np.int64)
    for i in range(n + 1):
        offsets[i + 1] = offsets[i] + len(compositions[indices[i] : indices[i + 1]])

    for i in range(n + 1):
        block = bihomogeneous_blocks[i]
        diag = np.diag(block)
        comps_i = compositions[indices[i] : indices[i + 1]]
        start = offsets[i]

        for j in range(len(comps_i)):
            alpha_fact = 1.0
            for value in comps_i[j]:
                alpha_fact *= factorials[int(value)]
            diagonal_coefficients[start + j] = (alpha_fact * diag[j]).real

    return diagonal_coefficients


@nb.njit(cache=True)
def _linear_form_power_coefficients(row, compositions, factorials):
    coeffs = np.zeros(len(compositions), dtype=np.complex128)

    for i, gamma in enumerate(compositions):
        degree = 0
        for x in gamma:
            degree += int(x)

        multinomial = factorials[degree]
        monomial = 1.0 + 0.0j

        for j, exponent in enumerate(gamma):
            exponent = int(exponent)
            multinomial /= factorials[exponent]
            if exponent != 0:
                monomial *= row[j] ** exponent

        coeffs[i] = multinomial * monomial

    return coeffs


@nb.njit(cache=True)
def _add_bihomogeneous_product_term(
    source_block: np.ndarray,
    add_block: np.ndarray,
    source_comps: np.ndarray,
    add_comps: np.ndarray,
    weight: float,
    out_block: np.ndarray,
) -> None:
    source_size = source_block.shape[0]
    add_size = add_block.shape[0]

    for i in range(source_size):
        for j in range(source_size):
            base = source_block[i, j]
            if base == 0.0:
                continue

            for gz in range(add_size):
                row = get_index_in_fock_subspace(source_comps[i] + add_comps[gz])
                cz = add_block[gz]

                for gw in range(add_size):
                    col = get_index_in_fock_subspace(source_comps[j] + add_comps[gw])
                    cw = np.conjugate(add_block[gw])

                    out_block[row, col] += weight * base * cz * cw


@nb.njit(cache=True)
def _probabilities_from_diagonal_coefficients(
    diagonals: np.ndarray,
    compositions: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    num_outcomes = outcomes.shape[0]
    num_alpha = compositions.shape[0]
    k = compositions.shape[1]

    probs = np.zeros(num_outcomes, dtype=np.complex128)

    for y_index in range(num_outcomes):
        total = 0.0 + 0.0j

        for a_index in range(num_alpha):
            alpha = compositions[a_index]

            ok = True
            sign_power = 0
            factor = 1.0

            for j in range(k):
                aj = int(alpha[j])
                yj = int(outcomes[y_index, j])

                if aj < yj:
                    ok = False
                    break

                sign_power += aj - yj
                factor *= comb(aj, yj)

            if ok:
                if sign_power % 2 == 1:
                    factor = -factor
                total += diagonals[a_index] * factor

        probs[y_index] = total

    return probs
