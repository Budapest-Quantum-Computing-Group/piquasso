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

import functools
from typing import Tuple

import numpy as np
import numba as nb

from piquasso._math.combinatorics import comb

from piquasso.api.config import Config
from piquasso._math.combinatorics import partitions
from piquasso._math.gradients import (
    create_single_mode_displacement_gradient,
    create_single_mode_squeezing_gradient,
)
from piquasso._math.gate_matrices import (
    create_single_mode_displacement_matrix,
    create_single_mode_squeezing_matrix,
)
from piquasso._math.indices import get_index_in_fock_space

from piquasso.api.connector import BaseConnector


@nb.njit(cache=True)
def cutoff_fock_space_dim(cutoff: int, d: int) -> int:
    r"""
    Calculates the dimension of the cutoff Fock space with the relation

    ..math::
        \sum_{i=0}^{c - 1} {d + i - 1 \choose i} = {d + c - 1 \choose d}.
    """
    return comb(d + cutoff - 1, d)


@nb.njit(cache=True)
def cutoff_fock_space_dim_array(cutoff, d):
    ret = np.empty(cutoff.shape, dtype=np.int32)

    for i in range(len(cutoff)):
        ret[i] = comb(d + cutoff[i] - 1, d)

    return ret


@nb.njit(cache=True)
def symmetric_subspace_cardinality(d: int, n: int) -> int:
    return comb(d + n - 1, n)


@nb.njit(cache=True)
def nb_get_fock_space_basis(d: int, cutoff: int) -> np.ndarray:
    size = cutoff_fock_space_dim(cutoff=cutoff, d=d)

    ret = np.empty((size, d), dtype=np.int32)
    current_row = 0
    for n in range(cutoff):
        num_rows = symmetric_subspace_cardinality(d, n)
        out = ret[current_row : current_row + num_rows, :]
        _ = partitions(boxes=d, particles=n, out=out)
        current_row += num_rows

    return ret


get_fock_space_basis = functools.lru_cache(maxsize=None)(nb_get_fock_space_basis)


def get_single_mode_squeezing_operator(
    r: float,
    phi: float,
    cutoff: int,
    complex_dtype: np.dtype,
    connector: BaseConnector,
) -> np.ndarray:
    @connector.custom_gradient
    def _single_mode_squeezing_operator(r, phi):
        r = connector.preprocess_input_for_custom_gradient(r)
        phi = connector.preprocess_input_for_custom_gradient(phi)

        matrix = create_single_mode_squeezing_matrix(
            r,
            phi,
            cutoff,
            complex_dtype=complex_dtype,
            connector=connector,
        )
        grad = create_single_mode_squeezing_gradient(
            r,
            phi,
            cutoff,
            matrix,
            connector,
        )
        return matrix, grad

    return _single_mode_squeezing_operator(r, phi)


def get_single_mode_cubic_phase_operator(
    gamma: float, cutoff: int, hbar: float, connector: BaseConnector
) -> np.ndarray:
    r"""Cubic Phase gate.

    The definition of the Cubic Phase gate is

    .. math::
        \operatorname{CP}(\gamma) = e^{i \hat{x}^3 \frac{\gamma}{3 \hbar}}

    The Cubic Phase gate transforms the annihilation operator as

    .. math::
        \operatorname{CP}^\dagger(\gamma) \hat{a} \operatorname{CP}(\gamma) =
            \hat{a} + i\frac{\gamma(\hat{a} +\hat{a}^\dagger)^2}{2\sqrt{2/\hbar}}

    It transforms the :math:`\hat{p}` quadrature as follows:

    .. math::
        \operatorname{CP}^\dagger(\gamma) \hat{p} \operatorname{CP}(\gamma) =
            \hat{p} + \gamma \hat{x}^2.

    Args:
        gamma (float): The Cubic Phase parameter.
        hbar (float): Scaling parameter.
    Returns:
        np.ndarray:
            The resulting transformation, which could be applied to the state.
    """

    np = connector.np

    annih = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    position = (annih.T + annih) * np.sqrt(hbar / 2)
    return connector.expm(1j * connector.powm(position, 3) * (gamma / (3 * hbar)))


def operator_basis(space):
    for index, basis in enumerate(space):
        for dual_index, dual_basis in enumerate(space):
            yield (index, dual_index), (basis, dual_basis)


def get_creation_operator(
    modes: Tuple[int, ...], space: np.ndarray, config: Config
) -> np.ndarray:
    d = len(space[0])
    size = cutoff_fock_space_dim(cutoff=config.cutoff, d=d)
    operator = np.zeros(shape=(size,) * 2, dtype=config.complex_dtype)

    for index, basis in enumerate(space):
        basis = np.array(basis)
        basis[modes,] += np.ones_like(modes)
        dual_index = get_index_in_fock_space(tuple(basis))

        if dual_index < size:
            operator[dual_index, index] = 1

    return operator


def get_annihilation_operator(
    modes: Tuple[int, ...], space: np.ndarray, config: Config
) -> np.ndarray:
    return get_creation_operator(modes, space, config).transpose()


def get_single_mode_displacement_operator(r, phi, cutoff, complex_dtype, connector):
    @connector.custom_gradient
    def _single_mode_displacement_operator(r, phi):
        r = connector.preprocess_input_for_custom_gradient(r)
        phi = connector.preprocess_input_for_custom_gradient(phi)

        matrix = create_single_mode_displacement_matrix(
            r,
            phi,
            cutoff,
            complex_dtype,
            connector=connector,
        )
        grad = create_single_mode_displacement_gradient(
            r,
            phi,
            cutoff,
            matrix,
            connector,
        )
        return matrix, grad

    return _single_mode_displacement_operator(r, phi)
