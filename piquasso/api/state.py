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

from typing import Tuple, Type, Optional

import abc
import copy

import numpy as np

from piquasso.api.config import Config


class State(abc.ABC):
    """The base class from which all `State` classes are derived.

    Properties:
        d (int): Instance attribute specifying the number of modes.
    """

    _config_class: Type[Config] = Config

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config.copy() if config is not None else self._config_class()

    def _get_auxiliary_modes(self, modes: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(np.delete(np.arange(self.d), modes))

    def copy(self) -> "State":
        """Returns an exact copy of this state.

        Returns:
            State: An exact copy of this state.
        """

        return copy.deepcopy(self)

    @property
    @abc.abstractmethod
    def d(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def fock_probabilities(self) -> np.ndarray:
        """Returns the particle detection probabilities.

        Note:
            The ordering of the Fock basis is increasing with particle numbers, and in
            each particle number conserving subspace, lexicographic ordering is used.

        Returns:
            numpy.ndarray: The particle detection probabilities.
        """
        pass

    @abc.abstractmethod
    def validate(self) -> None:
        """Validates the state."""
        pass

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
