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

import abc

from typing import Any, Tuple

import numpy

from piquasso.api.exceptions import NotImplementedCalculation


class BaseCalculator(abc.ABC):
    """The calculations for a simulation.

    NOTE: Every attribute of this class should be stateless!
    """

    np: Any
    fallback_np: Any

    def __deepcopy__(self, memo: Any) -> "BaseCalculator":
        """
        This method exists, because `copy.deepcopy` could not copy the entire modules
        and functions, and we don't need to, since every attribute of this class should
        be stateless.
        """

        return self

    def permanent(
        self, matrix: numpy.ndarray, rows: Tuple[int, ...], columns: Tuple[int, ...]
    ) -> float:
        raise NotImplementedCalculation()

    def hafnian(self, matrix: numpy.ndarray, reduce_on: Tuple[int, ...]) -> float:
        raise NotImplementedCalculation()

    def loop_hafnian(
        self, matrix: numpy.ndarray, diagonal: numpy.ndarray, reduce_on: Tuple[int, ...]
    ) -> float:
        raise NotImplementedCalculation()

    def assign(self, array, index, value):
        raise NotImplementedCalculation()

    def to_dense(self, index_map, dim):
        raise NotImplementedCalculation()

    def embed_in_identity(self, matrix, indices, dim):
        raise NotImplementedCalculation()

    def block(self, arrays):
        raise NotImplementedCalculation()

    def block_diag(self, *arrs):
        raise NotImplementedCalculation()

    def polar(self, matrix, side="right"):
        raise NotImplementedCalculation()

    def logm(self, matrix):
        raise NotImplementedCalculation()
