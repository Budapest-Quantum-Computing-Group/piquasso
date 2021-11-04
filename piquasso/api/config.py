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

from typing import Any

import os
import copy
import random
import numpy as np


class Config:
    """The configuration for the simulation."""

    def __init__(
        self,
        *,
        seed_sequence: Any = None,
        cache_size: int = 32,
        hbar: float = 2.0,
        use_torontonian: bool = False,
        cutoff: int = 4,
        measurement_cutoff: int = 5,
    ):
        self.seed_sequence = seed_sequence or int.from_bytes(
            os.urandom(8), byteorder="big"
        )
        self.cache_size = cache_size
        self.hbar = hbar
        self.use_torontonian = use_torontonian
        self.cutoff = cutoff
        self.measurement_cutoff = measurement_cutoff

    @property
    def seed_sequence(self):
        """The seed sequence used to generate random numbers during the simulation."""

        return self._seed_sequence

    @seed_sequence.setter
    def seed_sequence(self, value: Any) -> None:
        self._seed_sequence = value
        self.rng = np.random.default_rng(self._seed_sequence)
        random.seed(self._seed_sequence)

    def copy(self) -> "Config":
        """Returns an exact copy of this config object.

        Returns:
            Config: An exact copy of this config object.
        """

        return copy.deepcopy(self)
