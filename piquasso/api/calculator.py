#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

from typing import Any, Callable

import scipy

import numpy as np

from piquasso._math.permanent import np_glynn_gray_permanent
from piquasso._math.hafnian import hafnian_with_reduction, loop_hafnian_with_reduction

from piquasso.api.typing import PermanentFunction, HafnianFunction, LoopHafnianFunction


def assign(array, index, value):
    array[index] = value

    return array


def to_dense(index_map, dim):
    embedded_matrix = np.zeros((dim,) * 2, dtype=complex)

    for index, value in index_map.items():
        embedded_matrix[index] = value

    return embedded_matrix


def embed_in_identity(matrix, indices, dim):
    embedded_matrix = np.identity(dim, dtype=complex)

    embedded_matrix[indices] = matrix

    return embedded_matrix


class Calculator:
    """The customizable calculations for a simulation.

    NOTE: Every attribute of this class should be stateless!
    """

    def __init__(
        self,
        np: Any = np,
        fallback_np: Any = np,
        block_diag: Callable = scipy.linalg.block_diag,
        assign: Callable = assign,
        block: Callable = np.block,
        to_dense: Callable = to_dense,
        embed_in_identity: Callable = embed_in_identity,
        logm=scipy.linalg.logm,
        polar=scipy.linalg.polar,
        sqrtm=scipy.linalg.sqrtm,
        svd=np.linalg.svd,
        expm=scipy.linalg.expm,
        permanent_function: PermanentFunction = np_glynn_gray_permanent,
        hafnian_function: HafnianFunction = hafnian_with_reduction,
        loop_hafnian_function: LoopHafnianFunction = loop_hafnian_with_reduction,
    ):
        self.np = np
        self.fallback_np = fallback_np
        self.block_diag = block_diag
        self.assign = assign
        self.block = block
        self.to_dense = to_dense
        self.embed_in_identity = embed_in_identity
        self.logm = logm
        self.polar = polar
        self.sqrtm = sqrtm
        self.svd = svd
        self.expm = expm
        self.permanent_function = permanent_function
        self.hafnian_function = hafnian_function
        self.loop_hafnian_function = loop_hafnian_function

    def __deepcopy__(self, memo: Any) -> "Calculator":
        """
        This method exists, because `copy.deepcopy` could not copy the entire `numpy`
        module, and we don't need to, every attribute of this class should be stateless.
        """

        return self
