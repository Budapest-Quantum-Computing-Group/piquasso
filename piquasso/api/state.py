#
# Copyright 2021 Budapest Quantum Computing Group
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
import copy
from typing import Tuple

import numpy as np

from piquasso.api.config import Config


class State(abc.ABC):
    """The base class from which all `*State` classes are derived.

    Properties:
        d (int): Instance attribute specifying the number of modes.
    """

    def __init__(self, config: Config = None) -> None:
        self._config = config.copy() if config is not None else Config()

    @property
    @abc.abstractmethod
    def d(self) -> int:
        pass

    def copy(self) -> "State":
        return copy.deepcopy(self)

    @staticmethod
    def _get_operator_index(modes: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note:
            For indexing of numpy arrays, see
            https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
        """

        transformed_columns = np.array([modes] * len(modes))
        transformed_rows = transformed_columns.transpose()

        return transformed_rows, transformed_columns

    def _get_auxiliary_modes(self, modes: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(np.delete(np.arange(self.d), modes))

    @staticmethod
    def _get_auxiliary_operator_index(
        modes: Tuple[int, ...], auxiliary_modes: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        auxiliary_rows = tuple(np.array([modes] * len(auxiliary_modes)).transpose())

        return auxiliary_rows, auxiliary_modes

    @abc.abstractmethod
    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        """
        Returns the particle number detection probability using the occupation number
        specified as a parameter.

        Args:
            occupation_number (tuple):
                Tuple of natural numbers representing the number of particles in each
                mode.

        Returns:
            float: The probability of detection.
        """
        pass
