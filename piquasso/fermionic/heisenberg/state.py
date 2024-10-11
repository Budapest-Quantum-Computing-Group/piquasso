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

from typing import Optional, TYPE_CHECKING

from piquasso._math.validations import all_in_interval
from piquasso._math.linalg import is_selfadjoint, is_skew_symmetric
from piquasso._math.transformations import (
    from_xxpp_to_xpxp_transformation_matrix,
)

from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso.api.state import State
from piquasso.api.config import Config
from piquasso.api.connector import BaseConnector

if TYPE_CHECKING:
    import numpy as np


class HeisenbergState(State):
    def __init__(
        self,
        d: int,
        connector: BaseConnector,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(connector=connector, config=config)

        self._d = d

        self._correlations = []

    @property
    def d(self):
        return self._d

    @property
    def fock_probabilities(self) -> "np.ndarray":
        raise NotImplementedError()

    def get_particle_detection_probability(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()
