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

from typing import Tuple

import numpy as np

from piquasso.api.exceptions import NotImplementedCalculation


class BaseCalculator(abc.ABC):
    """The calculations for a simulation.

    NOTE: Every attribute of this class should be stateless!
    """

    def permanent(
        self, matrix: np.ndarray, rows: Tuple[int, ...], columns: Tuple[int, ...]
    ) -> float:
        raise NotImplementedCalculation()

    def hafnian(self, matrix: np.ndarray, reduce_on: Tuple[int, ...]) -> float:
        raise NotImplementedCalculation()

    def loop_hafnian(
        self, matrix: np.ndarray, diagonal: np.ndarray, reduce_on: Tuple[int, ...]
    ) -> float:
        raise NotImplementedCalculation()
