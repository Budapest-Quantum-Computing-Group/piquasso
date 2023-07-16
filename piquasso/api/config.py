#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from typing import Optional, Any

import os
import copy
import random
import numpy as np

from piquasso.core import _mixins


class Config(_mixins.CodeMixin):
    """The configuration for the simulation."""

    def __init__(
        self,
        *,
        seed_sequence: Optional[Any] = None,
        cache_size: int = 32,
        hbar: float = 2.0,
        use_torontonian: bool = False,
        cutoff: int = 4,
        measurement_cutoff: int = 5,
        dtype: type = np.float64,
        normalize: bool = True,
    ):
        self._original_seed_sequence = seed_sequence
        self.seed_sequence = seed_sequence or int.from_bytes(
            os.urandom(8), byteorder="big"
        )
        self.cache_size = cache_size
        self.hbar = hbar
        self.use_torontonian = use_torontonian
        self.cutoff = cutoff
        self.measurement_cutoff = measurement_cutoff
        self.dtype = np.float64 if dtype is float else dtype
        self.normalize = normalize

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return False
        return (
            self._original_seed_sequence == other._original_seed_sequence
            and self.cache_size == other.cache_size
            and self.hbar == other.hbar
            and self.use_torontonian == other.use_torontonian
            and self.cutoff == other.cutoff
            and self.measurement_cutoff == other.measurement_cutoff
            and self.dtype == other.dtype
            and self.normalize == other.normalize
        )

    def _as_code(self) -> str:
        default_config = Config()
        non_default_params = dict()

        if self._original_seed_sequence != default_config._original_seed_sequence:
            non_default_params["seed_sequence"] = self._original_seed_sequence
        if self.cache_size != default_config.cache_size:
            non_default_params["cache_size"] = self.cache_size
        if self.hbar != default_config.hbar:
            non_default_params["hbar"] = self.hbar
        if self.use_torontonian != default_config.use_torontonian:
            non_default_params["use_torontonian"] = self.use_torontonian
        if self.cutoff != default_config.cutoff:
            non_default_params["cutoff"] = self.cutoff
        if self.measurement_cutoff != default_config.measurement_cutoff:
            non_default_params["measurement_cutoff"] = self.measurement_cutoff
        if self.dtype != default_config.dtype:
            non_default_params["dtype"] = "np." + self.dtype.__name__
        if self.normalize != default_config.normalize:
            non_default_params["normalize"] = self.normalize

        if len(non_default_params) == 0:
            return "pq.Config()"
        else:
            params_string = ", ".join(
                f"{key}={value}" for key, value in non_default_params.items()
            )
            return f"pq.Config({params_string})"

    @property
    def seed_sequence(self):
        """The seed sequence used to generate random numbers during the simulation."""

        return self._seed_sequence

    @seed_sequence.setter
    def seed_sequence(self, value: Any) -> None:
        self._seed_sequence = value
        self.rng = np.random.default_rng(self._seed_sequence)
        random.seed(self._seed_sequence)

    @property
    def complex_dtype(self):
        """Returns the complex precision depending on the dtype of the Config class"""

        if self.dtype is np.float64:
            return np.complex128

        return np.complex64

    def copy(self) -> "Config":
        """Returns an exact copy of this config object.

        Returns:
            Config: An exact copy of this config object.
        """

        return copy.deepcopy(self)
