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

r"""Exact joint marginal photon-number sampling for Boson Sampling.

Given a Fock input state :math:`\ket{\mathbf{r}}` with :math:`n = \sum_b r_b`
photons interfering on a unitary :math:`U`, this module samples the joint
photon-number distribution restricted to a subset
:math:`T = (\ell_1, \dots, \ell_k)` of output modes, in time polynomial in
:math:`n` for fixed :math:`k`, i.e., without enumerating the exponentially many
outcomes on the unmeasured modes.

The algorithm is based on the (multivariate) probability generating function

.. math::
    G(\mathbf{x})
    = \bra{\mathbf{r}} \Gamma(U)^\dagger
      \prod_{j=1}^{k} x_j^{\hat{n}_{\ell_j}} \Gamma(U) \ket{\mathbf{r}}
    = \sum_{\mathbf{y}} \Pr(\mathbf{y}) \, \mathbf{x}^{\mathbf{y}}.

Using the normally ordered expansion
:math:`x^{\hat{n}} = \sum_{p} \frac{(x-1)^p}{p!} a^{\dagger p} a^p` and the
identity :math:`\bra{r} e^{\beta a^\dagger} e^{\gamma a} \ket{r} =
L_r(-\beta\gamma)` (:math:`L_r` being the Laguerre polynomial of degree
:math:`r`), all normally ordered moments are encoded in the polynomial

.. math::
    P(\mathbf{z}, \mathbf{w})
    = \prod_{b} L_{r_b}\left( -h_b(\mathbf{z}) \, g_b(\mathbf{w}) \right),
    \qquad
    h_b(\mathbf{z}) = \sum_{j} U_{\ell_j b} z_j,
    \quad
    g_b(\mathbf{w}) = \sum_{j} \overline{U_{\ell_j b}} w_j,

whose *diagonal* coefficients :math:`P_{\alpha, \alpha}` (of
:math:`\mathbf{z}^\alpha \mathbf{w}^\alpha`) yield the probabilities via

.. math::
    \Pr(\mathbf{y})
    = \sum_{\alpha \geq \mathbf{y}} \alpha! \, P_{\alpha, \alpha}
      \prod_{j=1}^k \binom{\alpha_j}{y_j} (-1)^{\alpha_j - y_j}.

Only the coefficients with total degree :math:`|\alpha| \leq n` are nonzero, and
there are just :math:`\binom{n + k}{k}` of them. :math:`P` is built up as the
product of its per-input-mode factors directly in this compact basis of
compositions: each factor :math:`L_{r_b}(-h_b g_b)` is expanded once, and the
running product is convolved with it while every partial product is truncated to
total degree :math:`n` in the :math:`\mathbf{z}` powers and in the
:math:`\mathbf{w}` powers. The convolution is carried out as sparse
matrix multiplications on the composition index, so the full
:math:`(n+1)^{2k}` coefficient array is never materialized; the working set
stays at :math:`O\!\left(\binom{n+k}{k}^2\right)`.

Sampling is then done mode by mode via the chain rule

.. math::
    \Pr(y_1, \dots, y_k)
    = \Pr(y_1) \, \Pr(y_2 \mid y_1) \cdots \Pr(y_k \mid y_1, \dots, y_{k-1}),

where each conditional is read off the joint distribution recovered from the
diagonal coefficients by the signed binomial transform above. Shots that share a
prefix are advanced together, so the work scales with the number of distinct
outcomes rather than with the number of shots.

References:
    - S. Aaronson and A. Arkhipov, *The Computational Complexity of Linear
      Optics*, Theorem 12.6, https://arxiv.org/abs/1011.3245 (collision-free
      marginals via Gurvits's algorithm; the present method generalizes to
      collided inputs).
    - W. Roga and M. Takeoka, *Classical simulation of boson sampling with
      sparse output*, https://arxiv.org/abs/1904.05494 (use of k-mode
      marginals).
"""

from typing import Dict, List, Tuple

import numpy as np

from scipy import sparse
from scipy.special import comb, factorial


def generate_marginal_samples(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: Tuple[int, ...],
    shots: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """Samples the measured modes from the exact joint marginal distribution.

    The measured modes are sampled one at a time via the chain rule.
    """
    initial_state = np.asarray(initial_state, dtype=int)
    modes_array = np.asarray(modes, dtype=int)

    n = int(np.sum(initial_state))
    k = len(modes_array)

    compositions, composition_index = _compositions(k, n)

    diagonal_coefficients = _diagonal_coefficients(
        interferometer, initial_state, modes_array, compositions, composition_index, n
    )

    joint = _joint_distribution(diagonal_coefficients, compositions, k, n)

    return _sample_chain_rule(joint, compositions, k, n, shots, rng)


def _compositions(k: int, n: int) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    r"""All length-``k`` nonnegative integer vectors with sum :math:`\leq n`."""
    compositions: List[Tuple[int, ...]] = []

    def _recurse(prefix: List[int], remaining: int) -> None:
        if len(prefix) == k:
            compositions.append(tuple(prefix))
            return
        for value in range(remaining + 1):
            _recurse(prefix + [value], remaining - value)

    _recurse([], n)

    composition_array = np.array(compositions, dtype=int).reshape(-1, k)
    index = {composition: i for i, composition in enumerate(compositions)}

    return composition_array, index


def _diagonal_coefficients(
    interferometer: np.ndarray,
    initial_state: np.ndarray,
    modes: np.ndarray,
    compositions: np.ndarray,
    composition_index: Dict[Tuple[int, ...], int],
    n: int,
) -> np.ndarray:
    r"""The scaled diagonal coefficients :math:`\alpha! \, P_{\alpha, \alpha}`.

    Returns a vector indexed by ``compositions``. The moment polynomial is built
    as the truncated product of its per-input-mode Laguerre factors, entirely in
    the compact basis of compositions.
    """
    k = len(modes)

    number_of_compositions = len(compositions)

    occupied = np.where(initial_state > 0)[0]
    occupations = initial_state[occupied]

    submatrix = np.asarray(interferometer, dtype=np.complex128)[np.ix_(modes, occupied)]

    degrees = compositions.sum(axis=1)

    product = sparse.lil_matrix(
        (number_of_compositions, number_of_compositions), dtype=np.complex128
    )
    product[composition_index[(0,) * k], composition_index[(0,) * k]] = 1.0
    product = product.tocsr()

    for column in range(len(occupied)):
        product = _multiply_factor(
            product,
            submatrix[:, column],
            occupations[column],
            compositions,
            composition_index,
            degrees,
            k,
            n,
        )

    diagonal = product.diagonal().real

    factorials = np.array(
        [np.prod(factorial(composition)) for composition in compositions]
    )

    return diagonal * factorials


def _multiply_factor(
    product: sparse.csr_matrix,
    column: np.ndarray,
    occupation: int,
    compositions: np.ndarray,
    composition_index: Dict[Tuple[int, ...], int],
    degrees: np.ndarray,
    k: int,
    n: int,
) -> sparse.csr_matrix:
    r"""Multiplies the running product by one Laguerre factor, truncated.

    The factor :math:`L_r(-h(\mathbf{z}) g(\mathbf{w}))` is a sum over its degree
    :math:`m` of a rank-one term in the :math:`\mathbf{z}` and :math:`\mathbf{w}`
    coefficients, so its action on the running product is a sum of
    ``Sz @ product @ Sw.T`` with sparse shift operators built once per degree.
    """
    number_of_compositions = len(compositions)

    result = sparse.csr_matrix(
        (number_of_compositions, number_of_compositions), dtype=np.complex128
    )

    laguerre_coefficients = [
        comb(occupation, m) * (-1.0) ** m / factorial(m) for m in range(occupation + 1)
    ]

    for m, laguerre_coefficient in enumerate(laguerre_coefficients):
        degree_m_indices = np.where(degrees == m)[0]

        z_weights = np.zeros(number_of_compositions, dtype=np.complex128)
        w_weights = np.zeros(number_of_compositions, dtype=np.complex128)

        for i in degree_m_indices:
            composition = compositions[i]
            multinomial = factorial(m)
            for value in composition:
                multinomial /= factorial(value)
            z_weights[i] = multinomial * np.prod(column**composition)
            w_weights[i] = multinomial * np.prod(np.conj(column) ** composition)

        shift_z = _shift_operator(z_weights, compositions, composition_index, n)
        shift_w = _shift_operator(w_weights, compositions, composition_index, n)

        result = result + laguerre_coefficient * (-1.0) ** m * (
            shift_z @ product @ shift_w.T
        )

    return result


def _shift_operator(
    weights: np.ndarray,
    compositions: np.ndarray,
    composition_index: Dict[Tuple[int, ...], int],
    n: int,
) -> sparse.csr_matrix:
    r"""The sparse operator that convolves one axis with ``weights``.

    Entry ``S[new, i]`` is :math:`\sum_p w_p` over those ``p`` with
    ``compositions[i] + compositions[p] == new`` and total degree at most ``n``.
    """
    number_of_compositions = len(compositions)

    nonzero = np.nonzero(weights)[0]

    rows: List[int] = []
    columns: List[int] = []
    data: List[complex] = []

    for i in range(number_of_compositions):
        base = compositions[i]
        for p in nonzero:
            shifted = base + compositions[p]
            if shifted.sum() <= n:
                rows.append(composition_index[tuple(shifted)])
                columns.append(i)
                data.append(weights[p])

    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(number_of_compositions, number_of_compositions),
        dtype=np.complex128,
    )


def _joint_distribution(
    diagonal_coefficients: np.ndarray,
    compositions: np.ndarray,
    k: int,
    n: int,
) -> np.ndarray:
    r"""Recovers the joint probabilities over ``compositions``.

    Applies the signed binomial transform
    :math:`\Pr(\mathbf{y}) = \sum_{\alpha \geq \mathbf{y}} c_\alpha
    \prod_j \binom{\alpha_j}{y_j} (-1)^{\alpha_j - y_j}` to the scaled diagonal
    coefficients :math:`c_\alpha`.
    """
    number_of_compositions = len(compositions)

    probabilities = np.zeros(number_of_compositions, dtype=np.float64)

    for i in range(number_of_compositions):
        y = compositions[i]
        dominating = np.all(compositions >= y, axis=1)
        alphas = compositions[dominating]

        signs = np.prod(
            comb(alphas, y) * (-1.0) ** (alphas - y),
            axis=1,
        )

        probabilities[i] = np.sum(diagonal_coefficients[dominating] * signs)

    return probabilities


def _sample_chain_rule(
    joint: np.ndarray,
    compositions: np.ndarray,
    k: int,
    n: int,
    shots: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """Draws ``shots`` samples mode by mode from the joint distribution.

    Shots that share an already-sampled prefix are advanced together.
    """
    probabilities = np.clip(joint, 0.0, None)

    # Each group is (number_of_shots, mask_of_consistent_compositions, prefix).
    groups: List[Tuple[int, np.ndarray, Tuple[int, ...]]] = [
        (shots, np.ones(len(compositions), dtype=bool), ())
    ]

    for axis in range(k):
        next_groups: List[Tuple[int, np.ndarray, Tuple[int, ...]]] = []

        for count, mask, prefix in groups:
            axis_probabilities = np.zeros(n + 1, dtype=np.float64)
            np.add.at(axis_probabilities, compositions[mask, axis], probabilities[mask])

            axis_probabilities = axis_probabilities / axis_probabilities.sum()

            outcome_counts = rng.multinomial(count, axis_probabilities)

            for value, value_count in enumerate(outcome_counts):
                if value_count == 0:
                    continue
                next_mask = mask & (compositions[:, axis] == value)
                next_groups.append((int(value_count), next_mask, prefix + (value,)))

        groups = next_groups

    samples: List[Tuple[int, ...]] = []
    for count, _, prefix in groups:
        samples.extend([prefix] * count)

    rng.shuffle(samples)

    return samples


def marginal_strategy_is_preferred(
    number_of_particles: int,
    number_of_measured_modes: int,
    number_of_occupied_modes: int,
    number_of_modes: int,
    shots: int,
) -> bool:
    """Decides whether sampling from the exact marginal distribution is cheaper
    than generating full samples and discarding the unmeasured modes.

    The decision compares estimated worst-case floating point operation counts of
    the two strategies and errs on the side of the existing full sampler.
    """
    n = number_of_particles
    k = number_of_measured_modes
    s = max(number_of_occupied_modes, 1)

    number_of_coefficients = comb(n + k, k, exact=True)

    # Building the coefficient table convolves s factors, each a sparse product
    # over the coefficient basis; recovering the joint and sampling are lower
    # order. The squared term reflects the (alpha, beta) working set.
    marginal_flops = s * number_of_coefficients**2 + shots * k * (n + 1)

    # Each full Clifford & Clifford sample evaluates permanents whose dominant
    # cost scales like n * d * 2^n.
    sampler_flops = shots * n * number_of_modes * 2.0 ** min(n, 48)

    return marginal_flops < sampler_flops
