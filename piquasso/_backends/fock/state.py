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

from typing import Tuple, Generator, Any, Dict

import abc
import numpy as np

from piquasso.api.config import Config
from piquasso.api.state import State

from piquasso._math import fock


class BaseFockState(State, abc.ABC):
    def __init__(self, *, d: int, config: Config = None) -> None:
        super().__init__(config=config)

        self._space = fock.FockSpace(
            d=d,
            cutoff=self._config.cutoff,
        )

    @property
    def d(self) -> int:
        return self._space.d

    @property
    def norm(self) -> int:
        return sum(self.fock_probabilities)

    def _as_code(self) -> str:
        return f"pq.Q() | pq.{self.__class__.__name__}(d={self.d})"

    @abc.abstractmethod
    def _get_empty(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def nonzero_elements(self) -> Generator[Tuple[complex, tuple], Any, None]:
        pass

    @property
    @abc.abstractmethod
    def density_matrix(self) -> np.ndarray:
        """The density matrix of the state in terms of the Fock basis vectors."""
        pass

    @abc.abstractmethod
    def reduced(self, modes: Tuple[int, ...]) -> "BaseFockState":
        """Reduces the state to a subsystem corresponding to the specified modes."""
        pass

    @abc.abstractmethod
    def normalize(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the Fock state to a vacuum state.
        """
        pass

    @property
    @abc.abstractmethod
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        pass

    @abc.abstractmethod
    def quadratures_mean_variance(
        self, modes: Tuple[int, ...], phi: float = 0
    ) -> Tuple[float, float]:
        """Calculates the mean and the variance of the quadratures of a Fock State"""
        pass
